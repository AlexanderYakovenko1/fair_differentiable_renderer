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
    return Scene({}, {
//                         std::make_shared<AxisAlignedRectangle>(0.31, 0.28, 0.25, 0.25, Color(RGBColor{0.7, 0.7, 0.0})),
//                         std::make_shared<Circle>(0.7, 0.7, 0.1, Color(RGBColor{0.543, 0.2232, 0.42})),
                         std::make_shared<Circle>(0.23, 0.72, 0.12, Color(RGBColor{0.1, 0.6, 1})),
                 }, 0, 1, 0, 1, RGBColor{0, 0, 0});
}

int main() {
    std::mt19937 rng(1337);


    Image<uint8_t> rgb_image(256, 256, 3);

    std::cout << "Rendering Scene1..." << std::flush;
    auto scene = Scene3();
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    scene.RenderToImage(rgb_image, rng, 10, 2e-3);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    Save8bitRgbImage("../scene1.png", rgb_image);
    std::cout << " Done" << std::endl;
    std::cout << "It took: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << " [µs]" << std::endl;


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