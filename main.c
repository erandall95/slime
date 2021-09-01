//https://thrust.github.io

//https://sagejenson.com/physarum
//https://uwe-repository.worktribe.com/output/980579
//https://github.com/fogleman/physarum

//compile with gcc -L/usr/local/lib -lSDL2 main.c -o physarum -framework OpenCL

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// #define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CPP //needed not to include <cuda_runtime_api.h>
// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>
// #include <thrust/generate.h>
// #include <thrust/sort.h>
// #include <thrust/copy.h>
// #include <algorithm>
// #include <cstdlib>

#include "SDL2/SDL.h" //use SDL for displaying image buffer http://www.libsdl.org

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define WINDOW_TITLE "Physarum"
#define DEFAULT_WINDOW_HEIGHT 480
#define DEFAULT_WINDOW_WIDTH 640
int HEIGHT = DEFAULT_WINDOW_HEIGHT,
    WIDTH  = DEFAULT_WINDOW_WIDTH;

bool WINDOW_IN_FOCUS = true;

//ORDER OF OPERATION
//1. SENSE: each cell sense's it next move based on it's surroundings and CELL_SENSE_DISTANCE
//2. MOVE: each cell is moved A CELL_MOVEMENT_DISTANCE to a new location
//3. DEPOSIT: each cell leaves a colored 'trail' from it's last location to it's new locations
//4. DEFUSE: the surrounding 'trail' pixels are averaged out by CELL_DEFUSE_DISTANCE
//5. DECAY: a DECAY_VALUE of pixel color is removed from each pixel until it is black - rgba(0,0,0,255)

//KERNAL OPERATION DESCRIPTION
//CELL[SENSE]:  find 'trail' of GOOD_TRAIL_AGE within CELL_SENSE_DISTANCE and CELL_SENSE_ANGLE based on last heading
//              then set heading in that direction plus "random" CELL_CHANCE_TO_IGNORE_TRAIL,
//              if there is a trail on more than one side, a random direction is chosen
//              else a random direction is chosen within that heading
//CELL[MOVE]:   move the cell location from current location CELL_MOVEMENT_DISTANCE pixels ahead based on heading
//CELL[DEPOSIT]:from last location to current location, place the cell color deposit
//              **A cell does not deposit infront of, to the left of, or to the right of itself, 
//              only behind the CELL_SENSE_ANGLE and heading. This effect considers the CELL_DEFUSE_DISTANCE
//CELL[DEFUSE]: All pixels along the 'trail' are diffuced outward by CELL_DEFUSE_DISTANCE
//DECAY:    All pixel colors are reduced by DECAY_VALUE

#define NUMBER_OF_CELLS_IN_SIMULATION 15000000

#define CELL_SENSE_DISTANCE 10 //pixels
#define CELL_SENSE_ANGLE 90.0F //degrees from cell center
#define CELL_CHANCE_TO_IGNORE_TRAIL 5 //0-100, where 0 is 0% and 100 is 100%
#define CELL_MOVE_DISTANCE 10 //pixels must be >= 1 NOTE: lower values seem to get stuck in corners
#define CELL_DEFUSE_DISTANCE 1 //pixels
#define DECAY_VALUE 3 //amount each pixel value is reduced by per frame update must be >= 1

typedef struct __attribute__ ((packed)) _coord {
    cl_int x;
    cl_int y;
} _coord;

//heading is randomly initialized to some valid angle in degrees
typedef struct __attribute__ ((packed)) _cell {
    _coord pos;
    cl_float heading;
    cl_uint red;
    cl_uint grn;
    cl_uint blu;
    //used for RNG
    cl_uint2 seed;
    //may make these values cell variable, but for now will just follow the defines
    cl_uint sense_distance;
    cl_float sense_angle;
    cl_uint chance_to_ignore_trail;
    cl_uint cell_move_distance;
    cl_uint cell_defuse_distance;
} _cell;

int parallel_compute(uint32_t* buffer, uint num_cells, _cell* cells);

uint degree(int d) {
    int max = 360;
    d = d % max;
    if(d < 0) {
        d = d + max;
    }
    return d;
}

bool boundary_check(_coord* p, uint max_x, uint max_y) {
    bool oob = false;
    if(p->x <= 0) {
        p->x = 1;
        // p->y = max_x + (p->x);
        oob = true;
    }
    if(p->y <= 0) {
        p->y = 1;
        // p->y = max_y + (p->y);
        oob = true;
    }
    if(p->x >= max_x) {
        p->x = max_x - 1;
        // p->x = 0 + (p->x - max_x);
        oob = true;
    }
    if(p->y >= max_y) {
        p->y = max_y - 1;
        // p->y = 0 + (p->y - max_y);
        oob = true;
    }
    return oob;
}

void decay_trails(Uint32* buffer) {
    for(int i = 0; i < WIDTH; i++) {
        for(int j = 0; j < HEIGHT; j++) {
            uint r = (buffer[j * (WIDTH) + i] >> 24) & 0xFF0000FF;
            uint g = (buffer[j * (WIDTH) + i] >> 16) & 0x00FF00FF;
            uint b = (buffer[j * (WIDTH) + i] >> 8) & 0x000000FF;
            r -= DECAY_VALUE;
            g -= DECAY_VALUE;
            b -= DECAY_VALUE;
            if(r < 0 || r > 255) r = 0x00;
            if(g < 0 || g > 255) g = 0x00;
            if(b < 0 || b > 255) b = 0x00;
            uint rgb = (r << 24) | (g << 16) | (b << 8) | (0xFF);
            buffer[j * (WIDTH) + i] = rgb;
        }
    }
}

/* 
 * A utility function to calculate area of triangle formed by (x1, y1),
 * (x2, y2) and (x3, y3) 
 **/
float triGonArea(_coord a, _coord b, _coord c) {
   return fabs((a.x*(b.y-c.y) + b.x*(c.y-a.y)+ c.x*(a.y-b.y))/2.0);
}

/* 
 * A function to check whether point P(x, y) lies inside the triangle formed
 * by A(x1, y1), B(x2, y2) and C(x3, y3) 
 **/
bool isInsideTriGon(_coord a, _coord b, _coord c, _coord p) {
    /* Calculate area of triangle ABC */
    float A = triGonArea (a, b, c);
    /* Calculate area of triangle PBC */
    float A1 = triGonArea (p, b, c);
    /* Calculate area of triangle PAC */
    float A2 = triGonArea (a, p, c);
    /* Calculate area of triangle PAB */
    float A3 = triGonArea (a, b, p);
    /* Check if sum of A1, A2 and A3 is same as A */
    return (A == A1 + A2 + A3);
}

bool point_in_coord_list(_coord* list, uint list_length, uint x, uint y) {
    bool match = false;
    for(uint k = 0; k < list_length; k++) {
        if(list[k].x == x && list[k].y == y) {
            match = true;
            break;
        }
    }
    return match;
}

void cell_update(_cell* cell, SDL_Surface* canvas, Uint32* buffer) {
    // printf("Cell position - x: %i, y: %i\n", cell->pos.x, cell->pos.y);
    // printf("Heading: %f\n", cell->heading);
    bool ignore = ((rand() % 100) + 1) <= cell->chance_to_ignore_trail;
    // if(ignore) printf("Cell is ignoring algorithmic instict\n");
    //SENSE
    //get endpoints of triangular sense
    double radian_convert = 3.14158265/180;
    float left_sense_heading = degree(cell->heading - cell->sense_angle/2);
    _coord left_sense = {
        .x = cell->pos.x + cell->cell_move_distance*cos(left_sense_heading * radian_convert),
        .y = cell->pos.y + cell->cell_move_distance*sin(left_sense_heading * radian_convert),
    };
    //boundary check, don't want to go offscreen
    boundary_check(&left_sense, WIDTH, HEIGHT);
    // printf("Left heading: %f, left X: %i, left Y: %i\n",left_sense_heading, left_sense.x, left_sense.y);
    float right_sense_heading = degree(cell->heading + cell->sense_angle/2);
    _coord right_sense = {
        .x = cell->pos.x + cell->cell_move_distance*cos(right_sense_heading * radian_convert),
        .y = cell->pos.y + cell->cell_move_distance*sin(right_sense_heading * radian_convert),
    };
    //boundary check, don't want to go offscreen
    boundary_check(&right_sense, WIDTH, HEIGHT);
    // printf("Right heading: %f, left X: %i, left Y: %i\n",right_sense_heading, right_sense.x, right_sense.y);
    
    uint area = triGonArea(cell->pos, left_sense, right_sense);
    //point BC
    uint lr_start_x = left_sense.x <= right_sense.x ? left_sense.x : right_sense.x;
    uint lr_end_x = left_sense.x >= right_sense.x ? left_sense.x : right_sense.x;
    uint lr_start_y = left_sense.y <= right_sense.y ? left_sense.y : right_sense.y;
    uint lr_end_y = left_sense.y >= right_sense.y ? left_sense.y : right_sense.y;
    //point AB
    uint left_start_x = left_sense.x <= cell->pos.x ? left_sense.x : cell->pos.x;
    uint left_end_x = left_sense.x >= cell->pos.x ? left_sense.x : cell->pos.x;
    uint left_start_y = left_sense.y <= cell->pos.y ? left_sense.y : cell->pos.y;
    uint left_end_y = left_sense.y >= cell->pos.y ? left_sense.y : cell->pos.y;
    //point AC
    uint right_start_x = right_sense.x <= cell->pos.x ? right_sense.x : cell->pos.x;
    uint right_end_x = right_sense.x >= cell->pos.x ? right_sense.x : cell->pos.x;
    uint right_start_y = right_sense.y <= cell->pos.y ? right_sense.y : cell->pos.y;
    uint right_end_y = right_sense.y >= cell->pos.y ? right_sense.y : cell->pos.y;

    // printf("lrxs: %i, lrex: %i, lrsy: %i, lrey: %i\n", lr_start_x, lr_end_x, lr_start_y, lr_end_y);
    // printf("lxs: %i, lex: %i, lsy: %i, ley: %i\n", left_start_x, left_end_x, left_start_y, left_end_y);
    // printf("rxs: %i, rex: %i, rsy: %i, rey: %i\n", right_start_x, right_end_x, right_start_y, right_end_y);
    //get all points inside triangle of sense - 3 bounding boxes then check if point is in the triangle
    //poi is an index trakcing the number of pixels in the sensory range
    uint poi_ind = 0;
    _coord pois[area*2];
    
    for(uint i = lr_start_x; i < lr_end_x; i++) {
        for(uint j = lr_start_y; j < lr_end_y; j++) {
            _coord pixel = {
                .x = i,
                .y = j,
            };
            bool inside = isInsideTriGon(cell->pos, left_sense, right_sense, pixel);
            if( inside ) {
                if( !point_in_coord_list(pois, poi_ind, i, j) && cell->pos.x != i && cell->pos.y != j ) {
                    pois[poi_ind].x = i;
                    pois[poi_ind].y = j;
                    poi_ind++;
                }
                // buffer[j * WIDTH + i] = SDL_MapRGBA(canvas->format, rand()%255+1, rand()%255+1, rand()%255+1, 255);
            }
        }
    }
    for(uint i = left_start_x; i < left_end_x; i++) {
        for(uint j = left_start_y; j < left_end_y; j++) {
            _coord pixel = {
                .x = i,
                .y = j,
            };
            bool inside = isInsideTriGon(cell->pos, left_sense, right_sense, pixel);
            if( inside ) {
                if( !point_in_coord_list(pois, poi_ind, i, j) && cell->pos.x != i && cell->pos.y != j ) {
                    pois[poi_ind].x = i;
                    pois[poi_ind].y = j;
                    poi_ind++;
                }
                // buffer[j * WIDTH + i] = SDL_MapRGBA(canvas->format, rand()%255+1, rand()%255+1, rand()%255+1, 255);
            }
        }
    }
    for(uint i = right_start_x; i < right_end_x; i++) {
        for(uint j = right_start_y; j < right_end_y; j++) {
            _coord pixel = {
                .x = i,
                .y = j,
            };
            bool inside = isInsideTriGon(cell->pos, left_sense, right_sense, pixel);
            if( inside ) {
                if( !point_in_coord_list(pois, poi_ind, i, j) && cell->pos.x != i && cell->pos.y != j ) {
                    pois[poi_ind].x = i;
                    pois[poi_ind].y = j;
                    poi_ind++;
                }
                // buffer[j * WIDTH + i] = SDL_MapRGBA(canvas->format, rand()%255+1, rand()%255+1, rand()%255+1, 255);
            }
        }
    }

    // buffer[cell.pos.y * WIDTH + cell.pos.x] = SDL_MapRGBA(canvas->format, 255, 0, 0, 255);
    // buffer[left_sense.y * WIDTH + left_sense.x] = SDL_MapRGBA(canvas->format, 255, 0, 0, 255);
    // buffer[right_sense.y * WIDTH + right_sense.x] = SDL_MapRGBA(canvas->format, 255, 0, 0, 255);

    //if the rgb in the buffer is lower than the cell's own rgb value,
    //add DECAY_VALUE to that color until any of the buffer rgb is at the same value of the cell's rgb
    _coord desired_point = {0, 0};
    uint num_desired_points = 0;
    for(uint i = 0; i < poi_ind; i++) {
        uint r = (buffer[pois[i].y * (WIDTH) + pois[i].x] >> 24) & 0xFF0000FF;
        uint g = (buffer[pois[i].y * (WIDTH) + pois[i].x] >> 16) & 0x00FF00FF;
        uint b = (buffer[pois[i].y * (WIDTH) + pois[i].x] >> 8) & 0x000000FF;
        //check if pixle is a scaled (decayed) value of the same cell color
        int r_diff = cell->red - r;
        bool r_scale = cell->red == r || (r_diff % DECAY_VALUE == 0 && r_diff >= 0 && r != 0);
        int g_diff = cell->grn - g;
        bool g_scale = cell->grn == g || (g_diff % DECAY_VALUE == 0 && g_diff >= 0 && g != 0);
        int b_diff = cell->blu - b;
        bool b_scale = cell->blu == b || (b_diff % DECAY_VALUE == 0 && b_diff >= 0 && b != 0);
        //true if all aspects of the pixel color are a decayed value of the cell's own color
        if(r_scale && g_scale && b_scale) {
            desired_point.x += pois[i].x;
            desired_point.y += pois[i].y;
            num_desired_points++;
        }
    }
    if(desired_point.x > 0) desired_point.x /= num_desired_points;
    if(desired_point.y > 0) desired_point.y /= num_desired_points;
    // printf("Desire C.O.M. x: %i, y: %i ... num: %i\n", desired_point.x, desired_point.y, num_desired_points);
    //set the cell's new heading toward the center of the desired trail or random if no desired trail
    if(num_desired_points != 0) {
        // buffer[desired_point.y * WIDTH + desired_point.x] = SDL_MapRGBA(canvas->format, cell->red, cell->grn, cell->blu, 255);
        //calculate new heading along current pos to center of mass of desired trail
        float x = desired_point.x - cell->pos.x;
        float y = desired_point.y - cell->pos.y;
        // printf("Diff x: %0.1f, diff y: %0.1f\n", x, y);
        float adj_heading = 0.0;
        if(x!=0 && y!=0) adj_heading = degree(atan2( y, x ) * (180 / 3.14158265));
        // printf("Adjusting heading by %f\n", adj_heading);
        if(adj_heading != 0.0) cell->heading = adj_heading;
    } else {
        //can use ignore boolean as it will generate a random heading anyway and will run after the sense function
        // printf("Cell cannot sense a similar cell trail\n");
        ignore = true;
    }

    if(ignore) {
        float upper = left_sense_heading <= right_sense_heading ? right_sense_heading : left_sense_heading;
        float lower = left_sense_heading >= right_sense_heading ? right_sense_heading : left_sense_heading;
        float random = ((float) rand()) / (float) RAND_MAX;
        if(upper - cell->sense_angle != lower) {
            cell->heading = degree(lower - (random * cell->sense_angle));
        } else {
            float diff = upper - lower;
            float r = random * diff;
            cell->heading = degree(lower+r);
        }
    }
    float rad_cell_heading = cell->heading * radian_convert;
    //MOVE
    _coord old_pos = {
        .x = cell->pos.x,
        .y = cell->pos.y,
    };
    cell->pos.x = cell->pos.x + cell->cell_move_distance*cos(rad_cell_heading);
    cell->pos.y = cell->pos.y + cell->cell_move_distance*sin(rad_cell_heading);
    //out of bounds check. If out of bounds, rotate heading by [sensory angle] degrees
    bool was_oob = boundary_check(&cell->pos, WIDTH, HEIGHT);
    // printf("New cell position - x: %i, y: %i\n", cell->pos.x, cell->pos.y);
    if(was_oob) {
        cell->heading = degree(cell->heading + cell->sense_angle);
        // printf("Cell was pointing out of bounds. Moving heading to: %f\n", cell->heading);
    }
    // printf("New Heading: %f\n", cell->heading);
    //place pixel of the cell color. It will overwrite whatever color is there
    // if(buffer[cell->pos.y * WIDTH + cell->pos.x]) buffer[cell->pos.y * WIDTH + cell->pos.x] = SDL_MapRGBA(canvas->format, cell->red, cell->grn, cell->blu, 255);
    buffer[cell->pos.y * WIDTH + cell->pos.x] = SDL_MapRGBA(canvas->format, cell->red, cell->grn, cell->blu, 255);
    //DEPOSIT
    _coord line[cell->cell_move_distance];
    for(int i = 0; i < cell->cell_move_distance; i++) {
        line[i].x = old_pos.x + (i)*cos(rad_cell_heading);
        line[i].y = old_pos.y + (i)*sin(rad_cell_heading);
        boundary_check(&line[i], WIDTH, HEIGHT);
        // printf("Line %i - x: %i y: %i\n", i, line[i].x, line[i].y);
        buffer[line[i].y * WIDTH + line[i].x] = SDL_MapRGBA(canvas->format, cell->red, cell->grn, cell->blu, 255);
    }
    //DEFUSE
    //should not diffuse into the new sensory range [using isInsideTriGon()]
    left_sense_heading = degree(cell->heading - cell->sense_angle/2);
    left_sense.x = cell->pos.x + cell->cell_move_distance*cos(left_sense_heading * radian_convert);
    left_sense.y = cell->pos.y + cell->cell_move_distance*sin(left_sense_heading * radian_convert);
    //boundary check, don't want to go offscreen
    boundary_check(&left_sense, WIDTH, HEIGHT);
    // printf("New left heading: %f, left X: %i, left Y: %i\n",left_sense_heading, left_sense.x, left_sense.y);
    right_sense_heading = degree(cell->heading + cell->sense_angle/2);
    right_sense.x = cell->pos.x + cell->cell_move_distance*cos(right_sense_heading * radian_convert);
    right_sense.y = cell->pos.y + cell->cell_move_distance*sin(right_sense_heading * radian_convert);
    //boundary check, don't want to go offscreen
    boundary_check(&right_sense, WIDTH, HEIGHT);
    // printf("New right heading: %f, left X: %i, left Y: %i\n",right_sense_heading, right_sense.x, right_sense.y);
    //for every point in the area of a circle with radius CELL_DEFUSE_DISTANCE along the points of movement
    //place a pixel of the average surrounding pixels colors
    //also confirm the pixel is on screen and not in the sensory area
    for(int i = 0; i < cell->cell_move_distance; i++) {
        //box of DEFUSE DISTANCE around each point
        _coord start = {
            .x = line[i].x-cell->cell_defuse_distance,
            .y = line[i].y-cell->cell_defuse_distance,
        };
        _coord end = {
            .x = line[i].x+cell->cell_defuse_distance,
            .y = line[i].y+cell->cell_defuse_distance,
        };
        bool defuse_start_oob = boundary_check(&start, WIDTH, HEIGHT);
        bool defuse_end_oob = boundary_check(&end, WIDTH, HEIGHT);
        // if(defuse_start_oob || defuse_end_oob) printf("Trail defuse attempted to go out of bounds\n");
        _coord def_area[(end.x - start.x) * (end.y - start.y)];
        int def_area_p = 0;
        for(int j = start.x; j < end.x; j++) {
            for(int k = start.y; k < end.y; k++) {
                bool inbound_x = j >=0 && j<=WIDTH;
                bool inbound_y = k >=0 && k<=HEIGHT;
                _coord p = {
                    .x = j,
                    .y = k,
                };
                bool inside_sensory_area = isInsideTriGon(cell->pos, left_sense, right_sense, p);
                if( inbound_x && inbound_y && !inside_sensory_area ) {
                    buffer[p.y * WIDTH + p.x] = SDL_MapRGBA(canvas->format, cell->red, cell->grn, cell->blu, 255);
                    def_area[def_area_p] = p;
                    def_area_p++;
                }
            }
        }
        // for each point in the box, defuse
        for(uint p = 0; p < def_area_p; p++) {
            uint ar = 0;
            uint ag = 0;
            uint ab = 0;
            for(int m = -1; m <= 1; m++) {
                for(int n = -1; n <= 1; n++) {
                    //skip center
                    if(m == 0 && n == 0) continue;
                    //confirm valid index
                    _coord def_pos = {
                        .x = (def_area[p].x+n),
                        .y = (def_area[p].y+m),
                    };
                    bool x_in_bounds = def_pos.x+n <= WIDTH && def_pos.x+n >= 0;
                    bool y_in_bounds = def_pos.y+m < HEIGHT && def_pos.y+m >= 0;
                    bool inside_sensory_area = isInsideTriGon(cell->pos, left_sense, right_sense, def_pos);
                    if(x_in_bounds && y_in_bounds && !inside_sensory_area) {
                        ar += (buffer[(def_pos.y+m) * (WIDTH) + (def_pos.x+n)] >> 24) & 0xFF0000FF;
                        ag += (buffer[(def_pos.y+m) * (WIDTH) + (def_pos.x+n)] >> 16) & 0x00FF00FF;
                        ab += (buffer[(def_pos.y+m) * (WIDTH) + (def_pos.x+n)] >> 8) & 0x000000FF;
                    }
                }
            }
            ar /= 8;
            ag /= 8;
            ab /= 8;
            buffer[def_area[p].y * WIDTH + def_area[p].x] = SDL_MapRGBA(canvas->format, ar, ag, ab, 255);
        }
    }
    // printf("\n");
}

void tick(SDL_Window* window, SDL_Surface* surface, SDL_Surface* canvas, _cell* cells, uint num_cells) {
    Uint32 *buffer = (Uint32*) canvas->pixels;
    SDL_LockSurface(canvas);

    parallel_compute(buffer, num_cells, cells);

    // for(int i = 0; i < num_cells; i++) {
    //     cell_update(&cells[i], canvas, buffer);
    // }
    // decay_trails(buffer);
    // paint and update canvas
    SDL_UnlockSurface(canvas);
    SDL_BlitSurface(canvas, 0, surface, 0);
    //update the display
    SDL_UpdateWindowSurface(window);
}

void cell_factory(_cell* cells, uint start, uint num_cells_create, uint8_t r, uint8_t g, uint8_t b) {
    for(uint i = start; i < num_cells_create; i++) {
        cells[i].pos.x = rand() % WIDTH-1;
        cells[i].pos.y = rand() % HEIGHT-1;
        cells[i].heading = degree(rand() % 360);
        cells[i].red = r;
        cells[i].grn = g;
        cells[i].blu = b;
        // cells[i].seed = [0,0];
        cells[i].sense_distance = CELL_SENSE_DISTANCE;
        cells[i].sense_angle = CELL_SENSE_ANGLE;
        cells[i].chance_to_ignore_trail = CELL_CHANCE_TO_IGNORE_TRAIL;
        cells[i].cell_move_distance = CELL_MOVE_DISTANCE;
        cells[i].cell_defuse_distance = CELL_DEFUSE_DISTANCE;
    }
}

void sdl_print_window_event(const SDL_Event* event) {
    if (event->type == SDL_WINDOWEVENT) {
        switch (event->window.event) {
        case SDL_WINDOWEVENT_SHOWN:
            // SDL_Log("Window %d shown", event->window.windowID);
            break;
        case SDL_WINDOWEVENT_HIDDEN:
            // SDL_Log("Window %d hidden", event->window.windowID);
            break;
        case SDL_WINDOWEVENT_EXPOSED:
            // SDL_Log("Window %d exposed", event->window.windowID);
            break;
        case SDL_WINDOWEVENT_MOVED:
            // SDL_Log("Window %d moved to %d,%d",
            //         event->window.windowID, event->window.data1,
            //         event->window.data2);
            break;
        case SDL_WINDOWEVENT_RESIZED:
            // SDL_Log("Window %d resized to %dx%d",
            //         event->window.windowID, event->window.data1,
            //         event->window.data2);
            break;
        case SDL_WINDOWEVENT_SIZE_CHANGED:
            // SDL_Log("Window %d size changed to %dx%d",
            //         event->window.windowID, event->window.data1,
            //         event->window.data2);
            HEIGHT = event->window.data2;
            WIDTH = event->window.data1;
            break;
        case SDL_WINDOWEVENT_MINIMIZED:
            // SDL_Log("Window %d minimized", event->window.windowID);
            break;
        case SDL_WINDOWEVENT_MAXIMIZED:
            // SDL_Log("Window %d maximized", event->window.windowID);
            break;
        case SDL_WINDOWEVENT_RESTORED:
            // SDL_Log("Window %d restored", event->window.windowID);
            break;
        case SDL_WINDOWEVENT_ENTER:
            // SDL_Log("Mouse entered window %d",
            //         event->window.windowID);
            break;
        case SDL_WINDOWEVENT_LEAVE:
            // SDL_Log("Mouse left window %d", event->window.windowID);
            break;
        case SDL_WINDOWEVENT_FOCUS_GAINED:
            // SDL_Log("Window %d gained keyboard focus",
            //         event->window.windowID);
            WINDOW_IN_FOCUS = true;
            break;
        case SDL_WINDOWEVENT_FOCUS_LOST:
            // SDL_Log("Window %d lost keyboard focus",
            //         event->window.windowID);
            WINDOW_IN_FOCUS = false;
            break;
        case SDL_WINDOWEVENT_CLOSE:
            // SDL_Log("Window %d closed", event->window.windowID);
            break;
    #if SDL_VERSION_ATLEAST(2, 0, 5)
        case SDL_WINDOWEVENT_TAKE_FOCUS:
            // SDL_Log("Window %d is offered a focus", event->window.windowID);
            break;
        case SDL_WINDOWEVENT_HIT_TEST:
            // SDL_Log("Window %d has a special hit test", event->window.windowID);
            break;
    #endif
        default:
            SDL_Log("Window %d got unknown event %d",
                    event->window.windowID, event->window.event);
            break;
        }
    }
}

void get_window_resolution() {
    SDL_DisplayMode DM;
    SDL_GetCurrentDisplayMode(0, &DM);
    HEIGHT = DM.h/2;
    WIDTH = DM.w/2;
    printf("Screen resolution is: %ix by %iy\n", WIDTH, HEIGHT);
}

void clear_screen(SDL_Window* window, SDL_Surface* surface, SDL_Surface* canvas) {
    Uint32 *buffer = (Uint32*) canvas->pixels;
    SDL_LockSurface(canvas);
    //set all pixels to SDL_MapRGBA(canvas->format, 0, 0, 0, 255);
    for(int i = 0; i < WIDTH; i++) {
        for(int j = 0; j < HEIGHT; j++) {
            uint r = (buffer[j * (WIDTH) + i] >> 24) & 0xFF0000FF;
            uint g = (buffer[j * (WIDTH) + i] >> 16) & 0x00FF00FF;
            uint b = (buffer[j * (WIDTH) + i] >> 8) & 0x000000FF;
            r -= 255;
            g -= 255;
            b -= 255;
            if(r < 0 || r > 255) r = 0x00;
            if(g < 0 || g > 255) g = 0x00;
            if(b < 0 || b > 255) b = 0x00;
            uint rgb = (r << 24) | (g << 16) | (b << 8) | (0xFF);
            buffer[j * (WIDTH) + i] = rgb;
        }
    }
    SDL_UnlockSurface(canvas);
    SDL_BlitSurface(canvas, 0, surface, 0);
    //update the display
    SDL_UpdateWindowSurface(window);
}

void event_loop(SDL_Window* window) {
    bool quit = false;
    SDL_Event e;
    
    SDL_Surface* surface = SDL_GetWindowSurface(window);
    SDL_Surface *canvas = SDL_CreateRGBSurfaceWithFormat(
        0, WIDTH, HEIGHT, 32, SDL_PIXELFORMAT_RGBA8888
    );

    uint num_cells = 5000;
    _cell cells[num_cells];
    cell_factory(cells, 0, num_cells/2, 0, 255 ,200);
    cell_factory(cells, num_cells/2 + 1, num_cells, 200, 0 ,255);

    tick(window, surface, canvas, cells, num_cells);
    
    while(!quit) {
        if(WINDOW_IN_FOCUS) {
            // tick(window, surface, canvas, cells, num_cells);
            // clear_screen(window, surface, canvas);
            // printf("\n");
        }
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) {
                quit = true;
            } else {
                sdl_print_window_event(&e);
            }
        }
        SDL_Delay(16); //16ms delay ≈ 60 fps;
    }
}

// void measure_func_execute_time() {
//     clock_t t;
//     t = clock();
    
//     t = clock() - t;
//     double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds
//     printf("Execution time: %f \n", time_taken);
// }

int main() {
    printf("\033c\n");

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        SDL_Log("Unable to initialize SDL: %s", SDL_GetError());
        return 1;
    }

    get_window_resolution();

    SDL_Window* window = SDL_CreateWindow(
        WINDOW_TITLE,
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        WIDTH,
        HEIGHT,
        SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_BORDERLESS
    );

    if (window == NULL) {
        SDL_Log("Unable to create window: %s", SDL_GetError());
        return 1;
    }
    //seed the RNG with the current time
    srand(time(0));

    //blocking event loop
    event_loop(window);
    // Cleanup.
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}

int parallel_compute(uint32_t* buffer, uint num_cells, _cell* cells) {
    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;
 
    fp = fopen("kernel/cell_update_kernel.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(0x100000);
    source_size = fread( source_str, 1, 0x100000, fp);
    fclose( fp );
 
    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;   
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    printf("Getting platform ID returned: %i\n", ret);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
    printf("Getting device ID returned: %i\n", ret);
    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
    printf("Creating context returned: %i\n", ret);
    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    printf("Creating command queue returned: %i\n", ret);
    // Create memory buffers on the device for each vector 
    //__global uint* buffer, __global int* HEIGHT, __global int* WIDTH, __global _cell* cell, __global int* DEVAY_VALUE
    cl_mem buffer_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, (WIDTH * HEIGHT) * sizeof(uint32_t), NULL, &ret);
    cl_mem height_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &ret);
    cl_mem width_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &ret);
    cl_mem cell_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, num_cells * sizeof(_cell), NULL, &ret);
    cl_mem decay_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &ret);
    printf("Loading all buffers returned: %i\n", ret);
    // Copy the lists A and B to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, width_mem_obj, CL_TRUE, 0, sizeof(int), &WIDTH, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, height_mem_obj, CL_TRUE, 0, sizeof(int), &HEIGHT, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, decay_mem_obj, CL_TRUE, 0, sizeof(int), (int*)DECAY_VALUE, 0, NULL, NULL);
    printf("Enqueueing buffers returned: %i\n", ret);
    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
    printf("Creating program returned: %i\n", ret);
    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    printf("Building program returned: %i\n", ret);
    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "cell_update", &ret);
    printf("Creating kernel returned: %i\n", ret);
    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&buffer_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&height_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&width_mem_obj);
    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&cell_mem_obj);
    ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&decay_mem_obj);
    printf("Setting arguments returned: %i\n", ret);
    // Execute the OpenCL kernel on the list
    size_t global_item_size = num_cells; // Process the entire lists
    size_t local_item_size = 64; // Divide work items into groups of 64
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
    printf("Executing kernel returned: %i\n", ret);
    // Read the memory buffer on the device
    ret = clEnqueueReadBuffer(command_queue, buffer_mem_obj, CL_TRUE, 0, (WIDTH * HEIGHT) * sizeof(uint32_t), buffer, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(command_queue, cell_mem_obj, CL_TRUE, 0, num_cells * sizeof(_cell), cells, 0, NULL, NULL);
    printf("Reading buffers returned: %i\n", ret);
    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(buffer_mem_obj);
    ret = clReleaseMemObject(height_mem_obj);
    ret = clReleaseMemObject(width_mem_obj);
    ret = clReleaseMemObject(cell_mem_obj);
    ret = clReleaseMemObject(decay_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    return 0;
}