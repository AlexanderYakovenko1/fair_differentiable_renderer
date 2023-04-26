# Differentiable rendering
Task from the Forward and Inverse Rendering course at CMC MSU

## Done
### Main
- [x] **(3 points)** Non-differentiable rendering of meshes and SDFs.  
Render result: `task1.png`
- [x] **(6 points)** Differentiable SDF rendering.  
Reference images: `02_reference.png`, `ACDC_logo.jpg`  
Optimization results: `task2_primitives.png`, `task2_image.png`
- [x] **(6 points)** Mesh texture derivatives.  
Reference image: `0_3_reference.png`  
Optimization result: `task3_texture.png`
### Bonus
- [x] **(1 point)** High resolution texture optimization (32x32 -> 64x64). 4x time increase, MSE dropped from 0.00118 to 0.00098   
Reference image: `0_3_reference.png`    
Optimization result: `task3_hires_texture.png`
- [x] **(7 points)** Mesh edge sampling. Optimization is divided into two steps (geometry and texture).  
Reference image: `01_reference.png`  
Optimization results: `task3_edge_sampling.png`

**Total: 23 points**

## Build
```bash
# from project root

mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

## Run
Create progress image directories and run the program
```bash
# from project root

mkdir sdf_primitives_progress \
  sdf_image_progress \
  texture_progress \
  hires_texture_progress \
  edge_geom_progress \
  edge_color_progress

cd build
./diff_renderer  
```

Output should be similar to this:
```
=========== Task1: Meshes & Textures ===========
Rendering to ../task1.png... Done
It took: 8856793 [µs]
============= Task2: Primitive SDF =============
Average time for one iteration: 41719 [µs]
Rendering to ../task2_primitives.png... Done
It took: 304074 [µs]
MSE: 0.00142147
=============== Task2: SDF Image ===============
Average time for one iteration: 2918861 [µs]
Rendering to ../task2_image.png... Done
It took: 6333162 [µs]
MSE: 0.000678656
================ Task3: Texture ================
Average time for one iteration: 532656 [µs]
Rendering to ../task3_texture.png... Done
It took: 2457683 [µs]
MSE: 0.00117606
============= Task3: HiRes Texture =============
Average time for one iteration: 2030653 [µs]
Rendering to ../task3_hires_texture.png... Done
It took: 8475673 [µs]
MSE: 0.00101807
============= Task3: Edge Sampling =============
Average time for one iteration: 530731 [µs]
Average time for one iteration: 530969 [µs]
Rendering to ../task3_edge_sampling.png... Done
It took: 2429228 [µs]
MSE: 0.00192414
```
