#ifndef SDF_SCENE_H
#define SDF_SCENE_H

#include <utility>
#include <vector>
#include <map>
#include <memory>
#include "distance_functions.h"
#include "image.h"
#include "mesh.h"

// https://en.wikipedia.org/wiki/Smoothstep
double smoothstep(double edge0, double edge1, double x) {
    if (x < edge0)
        return 0;

    if (x >= edge1)
        return 1;

    // Scale/bias into [0..1] range
    x = (x - edge0) / (edge1 - edge0);

    return x * x * (3 - 2 * x);
}

class Scene {
    TriangleMesh mesh_;
    std::vector<std::shared_ptr<SDF>> objects_;             // all objects in the scene
    double x_min_, x_max_, y_min_, y_max_; // left right top bottom borders of scene
    RGBColor background_;
public:
    Scene(TriangleMesh mesh, const std::vector<std::shared_ptr<SDF>>& objects,
          double x_min, double x_max, double y_min, double y_max, RGBColor background)
     :
        mesh_(std::move(mesh)),
        objects_(objects),
        x_min_(x_min),
        x_max_(x_max),
        y_min_(y_min),
        y_max_(y_max),
        background_(background)
    {}

    template<typename pixel_type>
    void RenderToImage(Image<pixel_type>& image, std::mt19937& rng, int num_samples=10, double eps=1e-3) {
        std::uniform_real_distribution<double> dist(-0.5, 0.5);

        for (int i = 0; i < image.height(); ++i) {
            for (int j = 0; j < image.width(); ++j) {
                double y_ = y_min_ + double(i) / image.width() * (y_max_ - y_min_);
                double x_ = x_min_ + double(j) / image.height() * (x_max_ - x_min_);

                std::vector<RGBColor> colors;
                std::vector<double> weights;
                for (int trial = 0; trial < num_samples; ++trial) {
                    double x = x_ + dist(rng) / image.height();
                    double y = y_ + dist(rng) / image.width();

                    RGBColor color = background_;

                    if (mesh_.distance(x, y) < eps) {
                        // as per task description has a_i = 1
                        // therefore overrides any other color in pixel
                        color = mesh_.getColor(x, y);
                    }

                    for (const auto &object: objects_) {
                        double distance = object->distance(x, y);
                        RGBColor hit_color = object->getColor(x, y);
                        double alpha = smoothstep(0, eps, -distance);
                        color = MixColors(hit_color, color, alpha);
                    }

                    colors.push_back(color);
                    weights.push_back(1);
                }

                auto color = MixColors(colors, weights);

                image(i, j, 0) = std::clamp(255 * color.r, 0., 255.);
                image(i, j, 1) = std::clamp(255 * color.g, 0., 255.);
                image(i, j, 2) = std::clamp(255 * color.b, 0., 255.);
            }
        }
    }
};

#endif //SDF_SCENE_H
