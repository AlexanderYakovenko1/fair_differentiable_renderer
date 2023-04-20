#include <iostream>
#include <vector>
#include <chrono>
#include <random>

#include "src/image.h"
#include "src/distance_functions.h"
#include "src/scene.h"
#include "src/mesh.h"
#include "src/opt.h"


// Shows that meshes and textures are supported
Scene Scene1() {
    auto grass = Image<double>("../grass_1k.png", 255.);

    return Scene(TriangleMesh(
            {Vec2d(0.0, 0.0), Vec2d(0.5, 0.0), Vec2d(0.0, 0.5),
             Vec2d(0.5, 0.0), Vec2d(1.0, 0.0), Vec2d(0.5, 0.5),
             Vec2d(0.04, 0.51), Vec2d(0.37, 0.61), Vec2d(0.17, 0.94)},
            {Vec3i(0, 1, 2), Vec3i(3, 4, 5), Vec3i(6, 7, 8)},
            {Color(grass), Color(grass), Color(RGBColor{1, 0, 0})}),{}, 0, 1, 0, 1, RGBColor{0, 0, 0});
}

Scene Scene2() {
    return Scene({},{
        std::make_shared<AxisAlignedRectangle>(0.31, 0.28, 0.25, 0.25, Color(RGBColor{0.7, 0.7, 0.0})),
        std::make_shared<Circle>(0.7, 0.7, 0.1, Color(RGBColor{0.543, 0.2232, 0.42})),
        std::make_shared<Circle>(0.23, 0.72, 0.12, Color(RGBColor{0.1, 0.6, 1})),
    }, 0, 1, 0, 1, RGBColor{0, 0, 0});
}

Scene Scene3() {
    return Scene({}, { //0.6, 0.7, 0.1, 0.15 // 0.85, 0.3, 0.6
                         std::make_shared<AxisAlignedRectangle>(0.31, 0.28, 0.25, 0.25, Color(RGBColor{0.7, 0.7, 0.0})),
                         std::make_shared<Circle>(0.7, 0.7, 0.1, Color(RGBColor{0.543, 0.2232, 0.42})),
                         std::make_shared<Circle>(0.23, 0.72, 0.12, Color(RGBColor{0.1, 0.6, 1})),
                 }, 0, 1, 0, 1, RGBColor{0, 0, 0});
}

Scene Test() {
    return Scene({}, {
                         std::make_shared<AxisAlignedRectangle>(0.31, 0.28, 0.25, 0.25, Color(RGBColor{0.7, 0.7, 0.0})),
//                         std::make_shared<Circle>(0.7, 0.7, 0.1, Color(RGBColor{0.543, 0.2232, 0.42})),
//            std::make_shared<Circle>(0.5, 0.55, 0.5, Color(RGBColor{0.9, 0.2, 0.1})),
    }, 0, 1, 0, 1, RGBColor{0, 0, 0});
}

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
            for (int k = 0; k < result.height(); ++k) {
                auto diff = result(i, j, k) * mult_res - reference(i, j, k) * mult_ref;
                sum += diff * diff;
            }
        }
    }
    return sqrt(sum);
}

int main() {
    std::mt19937 rng(1337);

    Image<uint8_t> rgb_image(256, 256, 3);
    auto ref = Image<double>("../02_reference.png", 255.);

//    std::cout << "Rendering Scene1..." << std::flush;
    auto scene = Scene3();
//    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
//    auto grads = scene.RenderToImage(rgb_image, rng, 10, 2e-3, ref);
//    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
//    Save8bitRgbImage("../refernce_rect.png", rgb_image);
//    std::cout << " Done" << std::endl;
//    std::cout << "It took: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << " [µs]" << std::endl;
    double eps = 1e-5;
    auto src = AxisAlignedRectangle(0.5, 0.65, 0.3, 0.15, Color(RGBColor{0.85, 0.3, 0.6}));
    auto dx = AxisAlignedRectangle(0.5+eps, 0.65, 0.3, 0.15, Color(RGBColor{0.85, 0.3, 0.6}));
    auto dy = AxisAlignedRectangle(0.5, 0.65+eps, 0.3, 0.15, Color(RGBColor{0.85, 0.3, 0.6}));
    auto dw = AxisAlignedRectangle(0.5, 0.65, 0.3+eps, 0.15, Color(RGBColor{0.85, 0.3, 0.6}));
    auto dh = AxisAlignedRectangle(0.5, 0.65, 0.3, 0.15+eps, Color(RGBColor{0.85, 0.3, 0.6}));
    for (double x = 0.; x < 1.; x+=0.1) {
        for (double y = 0.; y < 1.; y+=0.1) {
            auto grad = src.Dparam(x, y);

            auto dist = src.distance(x, y);
            auto dx_diff = (dx.distance(x, y) - dist) / eps;
            auto dy_diff = (dy.distance(x, y) - dist) / eps;
            auto dw_diff = (dw.distance(x, y) - dist) / eps;
            auto dh_diff = (dh.distance(x, y) - dist) / eps;

            std::cout << "got ";
            PrintArray(grad);
            std::cout << "diff: ";
            PrintArray<double>({dx_diff, dy_diff, dw_diff, dh_diff});
            std::cout << std::endl;
        }
    }

//    exit(0);
    int n_iters = 1000;
    auto params = scene.params();
    std::vector<double> geom_params;
    std::vector<double> color_params;
    for (auto [geom, color] : params) {
        geom_params.insert(geom_params.end(), geom.begin(), geom.end());
        color_params.insert(color_params.end(), color.begin(), color.end());
    }

    Adam opt_geom(geom_params, 2e-3, 0.9, 0.9);
    Adam opt_color(color_params, 2e-3, 0.9, 0.9);
    for (int iter = 1; iter < n_iters; ++iter) {
        params = scene.params();
        geom_params.clear();
        color_params.clear();
        for (auto [geom, color] : params) {
            geom_params.insert(geom_params.end(), geom.begin(), geom.end());
            color_params.insert(color_params.end(), color.begin(), color.end());
        }


        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        auto grads = scene.RenderToImage(rgb_image, rng, 1, 2e-3, ref);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        Save8bitRgbImage("../out_rect/scene" + std::to_string(iter) + ".png", rgb_image);
        std::cout << " Done" << std::endl;
        std::cout << "It took: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << " [µs]" << std::endl;

        std::vector<double> geom_grads;
        std::vector<double> color_grads;
        for (auto [geom, color] : grads) {
            geom_grads.insert(geom_grads.end(), geom.begin(), geom.end());
            color_grads.insert(color_grads.end(), color.begin(), color.end());
        }

        std::cout << "geom params: ";
        PrintArray(geom_params);
        std::cout << "geom grads: ";
        PrintArray(geom_grads);
        std::cout << "color params: ";
        PrintArray(color_params);
        std::cout << "color grads: ";
        PrintArray(color_grads);

        opt_geom.step(geom_params, geom_grads, iter);
        opt_color.step(color_params, color_grads, iter);

        int geom_idx = 0;
        int color_idx = 0;
        for (auto& [geom, color] : params) {
            for (auto& param : geom) {
                param = geom_params[geom_idx++];
            }
            for (auto& col : color) {
                col = color_params[color_idx++];
            }
        }
        scene.updateScene(params);
    }
    std::cout << "MSE: " << calcMSE(rgb_image, ref) << std::endl;



//    double eps = 1e-5;
//    for (double x = -0.2; x < 1.2; x+=0.1) {
//        double val = smoothstep(0, 1, x);
//        double next = smoothstep(0, 1, x + eps);
//        std::cout << (next - val) / eps << " " << Dsmoothstep(0, 1, x) << std::endl;
//    }

//    std::vector<double> x = {30};
//    auto opt = Adam(x, 0.1);
//
//    for (int i = 0; i < 150; ++i) {
//        auto loss = (x[0] - 15) * (x[0] - 15);
//        auto grad = 2 * (x[0] - 15);
//        printf("x=%.5lf loss=%.5lf grad=%.5lf\n", x[0], loss, grad);
//
//        opt.step(x, {grad});
//
//    }

}