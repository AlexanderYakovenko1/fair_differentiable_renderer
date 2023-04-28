#include <iostream>
#include <vector>
#include <chrono>
#include <random>

#include "src/image.h"
#include "src/distance_functions.h"
#include "src/scene.h"
#include "src/mesh.h"
#include "src/opt.h"


template <class T>
void PrintArray(const std::vector<T>& to_print) {
    for (auto value : to_print) {
        std::cout << value << ' ';
    }
    std::cout << std::endl;
}

double calcMSE(const Image<uint8_t>& result, const Image<double>& reference, double mult_res = 1/255., double mult_ref = 1.) {
    double sum = 0.0;
    for (int i = 0; i < result.height(); ++i) {
        for (int j = 0; j < result.height(); ++j) {
            for (int k = 0; k < result.channels(); ++k) {
                auto diff = result(i, j, k) * mult_res - reference(i, j, k) * mult_ref;
                sum += diff * diff;
            }
        }
    }
    return sum / result.height() / result.height() / result.channels();
}

void renderScene(Scene& scene, Image<uint8_t>& rgb_image, std::mt19937& rng, const std::string& output_path="../scene.png") {
    std::cout << "Rendering to " << output_path << "..." << std::flush;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    scene.RenderToImage(rgb_image, rng, 10, 2e-3);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    Save8bitRgbImage(output_path, rgb_image);
    std::cout << " Done" << std::endl;
    std::cout << "It took: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << " [µs]" << std::endl;
}

double optimizeScene(Scene& scene, Image<uint8_t>& rgb_image, const Image<double>& ref, std::mt19937& rng,
                     int n_iters=100, const std::string& output_dir="../output",
                     double geom_lr=2e-3, double color_lr=2e-3,
                     bool save_progress=true, bool verbose=false) {
    auto params = scene.params();
    std::vector<double> geom_params;
    std::vector<double> color_params;
    for (auto [geom, color]: params) {
        geom_params.insert(geom_params.end(), geom.begin(), geom.end());
        color_params.insert(color_params.end(), color.begin(), color.end());
    }

    Adam opt_geom(geom_params, geom_lr, 0.9, 0.9);
    Adam opt_color(color_params, color_lr, 0.9, 0.9);

    long runtime = 0;
    for (int iter = 1; iter <= n_iters; ++iter) {
        params = scene.params();
        geom_params.clear();
        color_params.clear();
        for (auto [geom, color]: params) {
            geom_params.insert(geom_params.end(), geom.begin(), geom.end());
            color_params.insert(color_params.end(), color.begin(), color.end());
        }


        if (verbose) { std::cout << "Started iteration " << iter << std::endl; }

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        auto grads = scene.RenderToImage(rgb_image, rng, 1, 2e-3, ref);
        grads = scene.EdgeSampling(rng, ref);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        if (save_progress) {
            Save8bitRgbImage(output_dir + "/scene" + std::to_string(iter) + ".png", rgb_image);
        }
        if (verbose) {
            std::cout << " Done" << std::endl;
            std::cout << "It took: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
                      << " [µs]" << std::endl;
        }
        runtime += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

        std::vector<double> geom_grads;
        std::vector<double> color_grads;
        for (auto [geom, color]: grads) {
            geom_grads.insert(geom_grads.end(), geom.begin(), geom.end());
            color_grads.insert(color_grads.end(), color.begin(), color.end());
        }

        if (verbose) {
            std::cout << "geom params: ";
            PrintArray(geom_params);
            std::cout << "geom grads: ";
            PrintArray(geom_grads);
            std::cout << "color params: ";
            PrintArray(color_params);
            std::cout << "color grads: ";
            PrintArray(color_grads);
        }

        opt_geom.step(geom_params, geom_grads, iter);
        opt_color.step(color_params, color_grads, iter);

        int geom_idx = 0;
        int color_idx = 0;
        for (auto &[geom, color]: params) {
            for (auto &param: geom) {
                param = geom_params[geom_idx++];
            }
            for (auto &col: color) {
                col = color_params[color_idx++];
            }
        }
        scene.updateScene(params);
    }
    std::cout << "Average time for one iteration: " << runtime / n_iters << " [µs]" << std::endl;

    return calcMSE(rgb_image, ref);
}


// Shows that meshes and textures are supported
void Task1_Meshes_and_Textures(std::mt19937& rng) {
    std::cout << "=========== Task1: Meshes & Textures ===========" << std::endl;
    Image<uint8_t> rgb_image(256, 256, 3);

    auto grass = Image<double>("../grass_64.png", 255.);

    auto scene = Scene(TriangleMesh(
                {Vec2d(0.0, 0.0), Vec2d(0.5, 0.0), Vec2d(0.0, 0.5),
                 Vec2d(0.5, 0.0), Vec2d(1.0, 0.0), Vec2d(0.5, 0.5),
                 Vec2d(0.04, 0.51), Vec2d(0.37, 0.61), Vec2d(0.17, 0.94)},
                {Vec3i(0, 1, 2), Vec3i(3, 4, 5), Vec3i(6, 7, 8)},
                {Color(grass), Color(grass), Color(RGBColor{1, 0, 0})}),{},
                        0, 1, 0, 1, RGBColor{0, 0, 0});

    renderScene(scene, rgb_image, rng, "../task1.png");
}

void Task2_Differentiable_SDF_primitives(std::mt19937& rng) {
    std::cout << "============= Task2: Primitive SDF =============" << std::endl;
    Image<uint8_t> rgb_image(256, 256, 3);

    auto ref = Image<double>("../02_reference.png", 255.);

    auto scene = Scene({},{
            std::make_shared<AxisAlignedRectangle>(0.31, 0.28, 0.25, 0.25, Color(RGBColor{0.7, 0.7, 0.0})),
            std::make_shared<Circle>(0.7, 0.7, 0.1, Color(RGBColor{0.543, 0.2232, 0.42})),
            std::make_shared<Circle>(0.23, 0.72, 0.12, Color(RGBColor{0.1, 0.6, 1})),
    }, 0, 1, 0, 1, RGBColor{0, 0, 0});

    optimizeScene(scene, rgb_image, ref, rng, 500, "../sdf_primitives_progress");
    renderScene(scene, rgb_image, rng, "../task2_primitives.png");
    std::cout << "MSE: " << calcMSE(rgb_image, ref) << std::endl;
}

void Task2_Differentiable_SDF_image(std::mt19937& rng) {
    std::cout << "=============== Task2: SDF Image ===============" << std::endl;
    Image<uint8_t> rgb_image(1024, 1024, 3);

    auto ref = Image<double>("../logo_1k.png", 255.);

    auto scene = Scene({}, {
            std::make_shared<SDFImage>(96, 96, 0.5, 0.5, 1., Color(RGBColor{0.9, 0.9, 0.9})),
    }, 0, 1, 0, 1, RGBColor{0, 0, 0});

    optimizeScene(scene, rgb_image, ref, rng, 100, "../sdf_image_progress");
    renderScene(scene, rgb_image, rng, "../task2_image.png");
    std::cout << "MSE: " << calcMSE(rgb_image, ref) << std::endl;
}

void Task3_Texture_Derivatives(std::mt19937& rng) {
    std::cout << "================ Task3: Texture ================" << std::endl;
    Image<uint8_t> rgb_image(256, 256, 3);

    auto ref = Image<double>("../0_3_reference.png", 255.);

    auto tex1 = Image<double>(32, 32, 3);
    auto tex2 = Image<double>(32, 32, 3);
    for (int i = 0; i < tex1.size(); ++i) {
        tex1(0, 0, i) = 0.5;
        tex2(0, 0, i) = 0.5;
    }

    auto scene = Scene(TriangleMesh(
                               {Vec2d(0.0, 0.0), Vec2d(0.5, 0.0), Vec2d(0.0, 0.5),
                                Vec2d(0.5, 0.0), Vec2d(1.0, 0.0), Vec2d(0.5, 0.5),
                                Vec2d(0.0, 0.5), Vec2d(0.5, 0.5), Vec2d(0.0, 1.0)},
                               {Vec3i(0, 1, 2), Vec3i(3, 4, 5), Vec3i(6, 7, 8)},
                               {Color(tex1), Color(tex2), Color(RGBColor{1, 0, 0})}),{},
                       0, 1, 0, 1, RGBColor{0, 0, 0});

    optimizeScene(scene, rgb_image, ref, rng, 300, "../texture_progress");
    renderScene(scene, rgb_image, rng, "../task3_texture.png");
    std::cout << "MSE: " << calcMSE(rgb_image, ref) << std::endl;
}

void Task3_HiRes_Texture_Derivatives(std::mt19937& rng) {
    std::cout << "============= Task3: HiRes Texture =============" << std::endl;
    Image<uint8_t> rgb_image(256, 256, 3);

    auto ref = Image<double>("../0_3_reference.png", 255.);

    auto tex1 = Image<double>(64, 64, 3);
    auto tex2 = Image<double>(64, 64, 3);
    for (int i = 0; i < tex1.size(); ++i) {
        tex1(0, 0, i) = 0.5;
        tex2(0, 0, i) = 0.5;
    }

    auto scene = Scene(TriangleMesh(
                               {Vec2d(0.0, 0.0), Vec2d(0.5, 0.0), Vec2d(0.0, 0.5),
                                Vec2d(0.5, 0.0), Vec2d(1.0, 0.0), Vec2d(0.5, 0.5),
                                Vec2d(0.0, 0.5), Vec2d(0.5, 0.5), Vec2d(0.0, 1.0)},
                               {Vec3i(0, 1, 2), Vec3i(3, 4, 5), Vec3i(6, 7, 8)},
                               {Color(tex1), Color(tex2), Color(RGBColor{1, 0, 0})}),{},
                       0, 1, 0, 1, RGBColor{0, 0, 0});

    optimizeScene(scene, rgb_image, ref, rng, 300, "../hires_texture_progress");
    renderScene(scene, rgb_image, rng, "../task3_hires_texture.png");
    std::cout << "MSE: " << calcMSE(rgb_image, ref) << std::endl;
}

void Task3_Edge_Sampling(std::mt19937& rng) {
    std::cout << "============= Task3: Edge Sampling =============" << std::endl;
    Image<uint8_t> rgb_image(256, 256, 3);

    auto ref = Image<double>("../01_reference.png", 255.);

    auto tex1 = Image<double>(32, 32, 3);
    auto tex2 = Image<double>(32, 32, 3);
    for (int i = 0; i < tex1.size(); ++i) {
        tex1(0, 0, i) = 0.5;
        tex2(0, 0, i) = 0.5;
    }

    auto scene = Scene(TriangleMesh(
                               {Vec2d(0.07, 0.05), Vec2d(0.46, 0.047), Vec2d(0.06, 0.51),
                                Vec2d(0.45, 0.0), Vec2d(0.97, 0.09), Vec2d(0.41, 0.45),
                                Vec2d(0.0, 0.45), Vec2d(0.45, 0.51), Vec2d(0.0, 0.86)},
                               {Vec3i(0, 1, 2), Vec3i(3, 4, 5), Vec3i(6, 7, 8)},
                               {Color(tex1), Color(tex2), Color(RGBColor{1, 0, 0})}),
                       {std::make_shared<Circle>(0.7, 0.7, 0.1, Color(RGBColor{0.543, 0.2232, 0.42}))},
                       0, 1, 0, 1, RGBColor{0, 0, 0});

    optimizeScene(scene, rgb_image, ref, rng, 100, "../edge_geom_progress", 2e-3, 0);
    optimizeScene(scene, rgb_image, ref, rng, 100, "../edge_color_progress", 0, 2e-2);
    renderScene(scene, rgb_image, rng, "../task3_edge_sampling.png");
    std::cout << "MSE: " << calcMSE(rgb_image, ref) << std::endl;
}

int main() {
    std::mt19937 rng(1337);

    Task1_Meshes_and_Textures(rng);
    Task2_Differentiable_SDF_primitives(rng);
    Task2_Differentiable_SDF_image(rng);
    Task3_Texture_Derivatives(rng);
    Task3_HiRes_Texture_Derivatives(rng);
    Task3_Edge_Sampling(rng);
}