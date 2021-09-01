typedef struct __attribute__ ((packed)) _coord {
    int x;
    int y;
} _coord;

//heading is randomly initialized to some valid angle in degrees
typedef struct __attribute__ ((packed)) _cell {
    _coord pos;
    float heading;
    uint red;
    uint grn;
    uint blu;
    //used for RNG
    uint2 seed;
    //may make these values cell variable, but for now will just follow the defines
    uint sense_distance;
    float sense_angle;
    uint chance_to_ignore_trail;
    uint cell_move_distance;
    uint cell_defuse_distance;
} _cell;

uint degree(int d);
bool boundary_check(_coord* p, int max_x, int max_y);
float triGonArea(_coord a, _coord b, _coord c);
bool isInsideTriGon(_coord a, _coord b, _coord c, _coord p);
bool point_in_coord_list(_coord* list, uint list_length, int x, int y);

uint degree(int d) {
    int max = 360;
    d = d % max;
    if(d < 0) {
        d = d + max;
    }
    return d;
}

bool boundary_check(_coord* p, int max_x, int max_y) {
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

bool point_in_coord_list(_coord* list, uint list_length, int x, int y) {
    bool match = false;
    for(uint k = 0; k < list_length; k++) {
        if(list[k].x == x && list[k].y == y) {
            match = true;
            break;
        }
    }
    return match;
}

__kernel void cell_update(__global uint* buffer, __global int* HEIGHT, __global int* WIDTH, __global _cell* cell, __global int* DEVAY_VALUE) {
    // Get the index of the current element
    uint i = get_global_id(0);
    //generate random numbers using XORSHIFT
    enum { A = 4294883355U };
    uint x=(cell[i].seed).x, c=(cell[i].seed).y;  // Unpack the state
    uint res=x^c;                                   // Calculate the result
    uint hi=mul_hi(x,A);                            // Step the RNG
    x=x*A+c;                                        // a random 32 bit integer
    c=hi+(x<c);                                     // a random 32 bit integer
    cell[i].seed=(uint2)(x,c);                     // Pack the state back up
    //PERFORM ALL OPERATIONS
    bool ignore = ((c % 100) + 1) <= cell[i].chance_to_ignore_trail;
    // if(ignore) printf("Cell is ignoring algorithmic instict\n");
    //SENSE
    //get endpoints of triangular sense
    double radian_convert = 3.14158265/180;
    float left_sense_heading = degree(cell[i].heading - cell[i].sense_angle/2);
    _coord left_sense = {
        .x = cell[i].pos.x + cell[i].cell_move_distance*cos(left_sense_heading * radian_convert),
        .y = cell[i].pos.y + cell[i].cell_move_distance*sin(left_sense_heading * radian_convert),
    };
    //boundary check, don't want to go offscreen
    boundary_check(&left_sense, (*WIDTH), (*HEIGHT));
    // printf("Left heading: %f, left X: %i, left Y: %i\n",left_sense_heading, left_sense.x, left_sense.y);
    float right_sense_heading = degree(cell[i].heading + cell[i].sense_angle/2);
    _coord right_sense = {
        .x = cell[i].pos.x + cell[i].cell_move_distance*cos(right_sense_heading * radian_convert),
        .y = cell[i].pos.y + cell[i].cell_move_distance*sin(right_sense_heading * radian_convert),
    };
    //boundary check, don't want to go offscreen
    boundary_check(&right_sense, (*WIDTH), (*HEIGHT));
    // printf("Right heading: %f, left X: %i, left Y: %i\n",right_sense_heading, right_sense.x, right_sense.y);
    
    uint area = triGonArea(cell[i].pos, left_sense, right_sense);
    //point BC
    int lr_start_x = left_sense.x <= right_sense.x ? left_sense.x : right_sense.x;
    int lr_end_x = left_sense.x >= right_sense.x ? left_sense.x : right_sense.x;
    int lr_start_y = left_sense.y <= right_sense.y ? left_sense.y : right_sense.y;
    int lr_end_y = left_sense.y >= right_sense.y ? left_sense.y : right_sense.y;
    //point AB
    int left_start_x = left_sense.x <= cell[i].pos.x ? left_sense.x : cell[i].pos.x;
    int left_end_x = left_sense.x >= cell[i].pos.x ? left_sense.x : cell[i].pos.x;
    int left_start_y = left_sense.y <= cell[i].pos.y ? left_sense.y : cell[i].pos.y;
    int left_end_y = left_sense.y >= cell[i].pos.y ? left_sense.y : cell[i].pos.y;
    //point AC
    int right_start_x = right_sense.x <= cell[i].pos.x ? right_sense.x : cell[i].pos.x;
    int right_end_x = right_sense.x >= cell[i].pos.x ? right_sense.x : cell[i].pos.x;
    int right_start_y = right_sense.y <= cell[i].pos.y ? right_sense.y : cell[i].pos.y;
    int right_end_y = right_sense.y >= cell[i].pos.y ? right_sense.y : cell[i].pos.y;

    // printf("lrxs: %i, lrex: %i, lrsy: %i, lrey: %i\n", lr_start_x, lr_end_x, lr_start_y, lr_end_y);
    // printf("lxs: %i, lex: %i, lsy: %i, ley: %i\n", left_start_x, left_end_x, left_start_y, left_end_y);
    // printf("rxs: %i, rex: %i, rsy: %i, rey: %i\n", right_start_x, right_end_x, right_start_y, right_end_y);
    //get all points inside triangle of sense - 3 bounding boxes then check if point is in the triangle
    //poi is an index trakcing the number of pixels in the sensory range
    uint poi_ind = 0;
    _coord pois[area*2];
    
    for(int i = lr_start_x; i < lr_end_x; i++) {
        for(int j = lr_start_y; j < lr_end_y; j++) {
            _coord pixel = {
                .x = i,
                .y = j,
            };
            bool inside = isInsideTriGon(cell[i].pos, left_sense, right_sense, pixel);
            if( inside ) {
                if( !point_in_coord_list(pois, poi_ind, i, j) && cell[i].pos.x != i && cell[i].pos.y != j ) {
                    pois[poi_ind].x = i;
                    pois[poi_ind].y = j;
                    poi_ind++;
                }
                // buffer[j * (*WIDTH) + i] = SDL_MapRGBA(canvas->format, rand()%255+1, rand()%255+1, rand()%255+1, 255);
            }
        }
    }
    for(int i = left_start_x; i < left_end_x; i++) {
        for(int j = left_start_y; j < left_end_y; j++) {
            _coord pixel = {
                .x = i,
                .y = j,
            };
            bool inside = isInsideTriGon(cell[i].pos, left_sense, right_sense, pixel);
            if( inside ) {
                if( !point_in_coord_list(pois, poi_ind, i, j) && cell[i].pos.x != i && cell[i].pos.y != j ) {
                    pois[poi_ind].x = i;
                    pois[poi_ind].y = j;
                    poi_ind++;
                }
                // buffer[j * (*WIDTH) + i] = SDL_MapRGBA(canvas->format, rand()%255+1, rand()%255+1, rand()%255+1, 255);
            }
        }
    }
    for(int i = right_start_x; i < right_end_x; i++) {
        for(int j = right_start_y; j < right_end_y; j++) {
            _coord pixel = {
                .x = i,
                .y = j,
            };
            bool inside = isInsideTriGon(cell[i].pos, left_sense, right_sense, pixel);
            if( inside ) {
                if( !point_in_coord_list(pois, poi_ind, i, j) && cell[i].pos.x != i && cell[i].pos.y != j ) {
                    pois[poi_ind].x = i;
                    pois[poi_ind].y = j;
                    poi_ind++;
                }
                // buffer[j * (*WIDTH) + i] = SDL_MapRGBA(canvas->format, rand()%255+1, rand()%255+1, rand()%255+1, 255);
            }
        }
    }

    // buffer[cell.pos.y * (*WIDTH) + cell.pos.x] = SDL_MapRGBA(canvas->format, 255, 0, 0, 255);
    // buffer[left_sense.y * (*WIDTH) + left_sense.x] = SDL_MapRGBA(canvas->format, 255, 0, 0, 255);
    // buffer[right_sense.y * (*WIDTH) + right_sense.x] = SDL_MapRGBA(canvas->format, 255, 0, 0, 255);

    //if the rgb in the buffer is lower than the cell's own rgb value,
    //add (*DEVAY_VALUE) to that color until any of the buffer rgb is at the same value of the cell's rgb
    _coord desired_point = {0, 0};
    uint num_desired_points = 0;
    for(uint i = 0; i < poi_ind; i++) {
        uint r = (buffer[pois[i].y * ((*WIDTH)) + pois[i].x] >> 24) & 0xFF0000FF;
        uint g = (buffer[pois[i].y * ((*WIDTH)) + pois[i].x] >> 16) & 0x00FF00FF;
        uint b = (buffer[pois[i].y * ((*WIDTH)) + pois[i].x] >> 8) & 0x000000FF;
        //check if pixle is a scaled (decayed) value of the same cell color
        int r_diff = cell[i].red - r;
        bool r_scale = cell[i].red == r || (r_diff % (*DEVAY_VALUE) == 0 && r_diff >= 0 && r != 0);
        int g_diff = cell[i].grn - g;
        bool g_scale = cell[i].grn == g || (g_diff % (*DEVAY_VALUE) == 0 && g_diff >= 0 && g != 0);
        int b_diff = cell[i].blu - b;
        bool b_scale = cell[i].blu == b || (b_diff % (*DEVAY_VALUE) == 0 && b_diff >= 0 && b != 0);
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
        // buffer[desired_point.y * (*WIDTH) + desired_point.x] = SDL_MapRGBA(canvas->format, 255, 0, 0, 255);
        //calculate new heading along current pos to center of mass of desired trail
        float x = desired_point.x - cell[i].pos.x;
        float y = desired_point.y - cell[i].pos.y;
        // printf("Diff x: %0.1f, diff y: %0.1f\n", x, y);
        float adj_heading = 0.0;
        if(x!=0 && y!=0) adj_heading = degree(atan2( y, x ) * (180 / 3.14158265));
        // printf("Adjusting heading by %f\n", adj_heading);
        if(adj_heading != 0.0) cell[i].heading = adj_heading;
    } else {
        //can use ignore boolean as it will generate a random heading anyway and will run after the sense function
        // printf("Cell cannot sense a similar cell trail\n");
        ignore = true;
    }

    if(ignore) {
        // float upper = left_sense_heading <= right_sense_heading ? right_sense_heading : left_sense_heading;
        // float lower = left_sense_heading >= right_sense_heading ? right_sense_heading : left_sense_heading;
        // float random = ((float) x) / (float) 2147483647;
        // if(upper - cell[i].sense_angle != lower) {
        //     cell[i].heading = degree(lower - (random * cell[i].sense_angle));
        // } else {
        //     float diff = upper - lower;
        //     float r = random * diff;
        //     cell[i].heading = degree(lower+r);
        // }
    }
    float rad_cell_heading = cell[i].heading * radian_convert;
    //MOVE
    _coord old_pos = {
        .x = cell[i].pos.x,
        .y = cell[i].pos.y,
    };
    cell[i].pos.x = cell[i].pos.x + cell[i].cell_move_distance*cos(rad_cell_heading);
    cell[i].pos.y = cell[i].pos.y + cell[i].cell_move_distance*sin(rad_cell_heading);
    //out of bounds check. If out of bounds, rotate heading by [sensory angle] degrees
    bool was_oob = boundary_check((__global)&cell[i].pos, (*WIDTH), (*HEIGHT));
    // printf("New cell position - x: %i, y: %i\n", cell[i].pos.x, cell[i].pos.y);
    if(was_oob) {
        cell[i].heading = degree(cell[i].heading + cell[i].sense_angle);
        // printf("Cell was pointing out of bounds. Moving heading to: %f\n", cell[i].heading);
    }
    printf("New Heading: %f\n", cell[i].heading);
    //place pixel of the cell color. It will overwrite whatever color is there
    // if(buffer[cell[i].pos.y * (*WIDTH) + cell[i].pos.x]) buffer[cell[i].pos.y * (*WIDTH) + cell[i].pos.x] = SDL_MapRGBA(canvas->format, cell[i].red, cell[i].grn, cell[i].blu, 255);
    // buffer[cell[i].pos.y * (*WIDTH) + cell[i].pos.x] = SDL_MapRGBA(canvas->format, cell[i].red, cell[i].grn, cell[i].blu, 255);
    buffer[cell[i].pos.y * (*WIDTH) + cell[i].pos.x] = ( (cell[i].red << 24) | (cell[i].grn << 16) | (cell[i].blu << 8) | 0xFF );
    //DEPOSIT
    _coord line[cell[i].cell_move_distance];
    for(uint i = 0; i < cell[i].cell_move_distance; i++) {
        line[i].x = old_pos.x + (i)*cos(rad_cell_heading);
        line[i].y = old_pos.y + (i)*sin(rad_cell_heading);
        boundary_check(&line[i], (*WIDTH), (*HEIGHT));
        // printf("Line %i - x: %i y: %i\n", i, line[i].x, line[i].y);
        // buffer[line[i].y * (*WIDTH) + line[i].x] = SDL_MapRGBA(canvas->format, cell[i].red, cell[i].grn, cell[i].blu, 255);
        buffer[line[i].y * (*WIDTH) + line[i].x] = ( (cell[i].red << 24) | (cell[i].grn << 16) | (cell[i].blu << 8) | 0xFF );
    }
    //DEFUSE
    //should not diffuse into the new sensory range [using isInsideTriGon()]
    left_sense_heading = degree(cell[i].heading - cell[i].sense_angle/2);
    left_sense.x = cell[i].pos.x + cell[i].cell_move_distance*cos(left_sense_heading * radian_convert);
    left_sense.y = cell[i].pos.y + cell[i].cell_move_distance*sin(left_sense_heading * radian_convert);
    //boundary check, don't want to go offscreen
    boundary_check(&left_sense, (*WIDTH), (*HEIGHT));
    // printf("New left heading: %f, left X: %i, left Y: %i\n",left_sense_heading, left_sense.x, left_sense.y);
    right_sense_heading = degree(cell[i].heading + cell[i].sense_angle/2);
    right_sense.x = cell[i].pos.x + cell[i].cell_move_distance*cos(right_sense_heading * radian_convert);
    right_sense.y = cell[i].pos.y + cell[i].cell_move_distance*sin(right_sense_heading * radian_convert);
    //boundary check, don't want to go offscreen
    boundary_check(&right_sense, (*WIDTH), (*HEIGHT));
    // printf("New right heading: %f, left X: %i, left Y: %i\n",right_sense_heading, right_sense.x, right_sense.y);
    //for every point in the area of a circle with radius CELL_DEFUSE_DISTANCE along the points of movement
    //place a pixel of the average surrounding pixels colors
    //also confirm the pixel is on screen and not in the sensory area
    for(uint i = 0; i < cell[i].cell_move_distance; i++) {
        //box of DEFUSE DISTANCE around each point
        _coord start = {
            .x = line[i].x-cell[i].cell_defuse_distance,
            .y = line[i].y-cell[i].cell_defuse_distance,
        };
        _coord end = {
            .x = line[i].x+cell[i].cell_defuse_distance,
            .y = line[i].y+cell[i].cell_defuse_distance,
        };
        // bool defuse_start_oob = boundary_check(&start, (*WIDTH), (*HEIGHT));
        // bool defuse_end_oob = boundary_check(&end, (*WIDTH), (*HEIGHT));
        boundary_check(&start, (*WIDTH), (*HEIGHT));
        boundary_check(&end, (*WIDTH), (*HEIGHT));
        // if(defuse_start_oob || defuse_end_oob) printf("Trail defuse attempted to go out of bounds\n");
        _coord def_area[(end.x - start.x) * (end.y - start.y)];
        int def_area_p = 0;
        for(int j = start.x; j < end.x; j++) {
            for(int k = start.y; k < end.y; k++) {
                bool inbound_x = j >=0 && j <= (*WIDTH);
                bool inbound_y = k >=0 && k <= (*HEIGHT);
                _coord p = {
                    .x = j,
                    .y = k,
                };
                bool inside_sensory_area = isInsideTriGon(cell[i].pos, left_sense, right_sense, p);
                if( inbound_x && inbound_y && !inside_sensory_area ) {
                    // buffer[p.y * (*WIDTH) + p.x] = SDL_MapRGBA(canvas->format, cell[i].red, cell[i].grn, cell[i].blu, 255);
                    buffer[p.y * (*WIDTH) + p.x] = ( (cell[i].red << 24) | (cell[i].grn << 16) | (cell[i].blu << 8) | 0xFF );
                    def_area[def_area_p] = p;
                    def_area_p++;
                }
            }
        }
        // for each point in the box, defuse
        for(int p = 0; p < def_area_p; p++) {
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
                    bool x_in_bounds = def_pos.x+n <= (*WIDTH) && def_pos.x+n >= 0;
                    bool y_in_bounds = def_pos.y+m < (*HEIGHT) && def_pos.y+m >= 0;
                    bool inside_sensory_area = isInsideTriGon(cell[i].pos, left_sense, right_sense, def_pos);
                    if(x_in_bounds && y_in_bounds && !inside_sensory_area) {
                        ar += (buffer[(def_pos.y+m) * ((*WIDTH)) + (def_pos.x+n)] >> 24) & 0xFF0000FF;
                        ag += (buffer[(def_pos.y+m) * ((*WIDTH)) + (def_pos.x+n)] >> 16) & 0x00FF00FF;
                        ab += (buffer[(def_pos.y+m) * ((*WIDTH)) + (def_pos.x+n)] >> 8) & 0x000000FF;
                    }
                }
            }
            ar /= 8;
            ag /= 8;
            ab /= 8;
            // buffer[def_area[p].y * (*WIDTH) + def_area[p].x] = SDL_MapRGBA(canvas->format, ar, ag, ab, 255);
            buffer[def_area[p].y * (*WIDTH) + def_area[p].x] = ( (ar << 24) | (ag << 16) | (ab << 8) | 0xFF );
        }
    }

}