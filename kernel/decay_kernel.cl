__kernel void decay(__global uint* flattened_canvas, __global uint *height, __global int *width, __global uint* decay_value) {
    
    // Get the index of the current element
    uint i = get_global_id(0);
    uint j = get_global_id(1);

    // Do the operation
    uint r = (flattened_canvas[j * (*width) + i] >> 24) & 0xFF0000FF;
    uint g = (flattened_canvas[j * (*width) + i] >> 16) & 0x00FF00FF;
    uint b = (flattened_canvas[j * (*width) + i] >> 8) & 0x0000FFFF;
    r -= *decay_value;
    g -= *decay_value;
    b -= *decay_value;
    if(r < 0 || r > 255) r = 0x00;
    if(g < 0 || g > 255) g = 0x00;
    if(b < 0 || b > 255) b = 0x00;
    uint rgb = (r << 24) | (g << 16) | (b << 8) | (0xFF);
    flattened_canvas[j * (*width) + i] = rgb;
}