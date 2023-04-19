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

double Dsmoothstep(double edge0, double edge1, double x) {
    if (x < edge0)
        return 0;

    if (x >= edge1)
        return 0;

    // Scale/bias into [0..1] range
    x = (x - edge0) / (edge1 - edge0);

    return (6 * x - 6 * x * x) / (edge1 - edge0);
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
    std::vector<std::vector<double>> RenderToImage(Image<pixel_type>& image, std::mt19937& rng, int num_samples=10, double eps=1e-3,
                                                   const Image<double>& reference = Image<double>()) {
        // for each obect in the scene stores geometrical and color differentials
        std::vector<std::pair<std::vector<double>,std::vector<double>>> Dscene;

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

                    std::vector<double> alphas;  // alpha mixing coefficients of final color in sample
                    std::vector<std::vector<double>> Dparams;
                    std::vector<std::vector<double>> Dcolors;
                    std::vector<double> Dedges;
                    RGBColor color = background_;

                    /// MESH

                    if (mesh_.distance(x, y) < eps) {
                        // as per task description has a_i = 1
                        // therefore overrides any other color in pixel
                        color = mesh_.getColor(x, y);
                        alphas.push_back(1);
                    } else {
                        alphas.push_back(0);
                    }
                    Dedges.push_back(0); // step func, edge sampling goes here i guess
                    Dparams.push_back(mesh_.Dparam(x, y));
                    Dcolors.push_back(mesh_.Dcolor(x, y));

                    /// END MESH


                    /// SDF

                    for (const auto &object: objects_) {
                        double distance = object->distance(x, y);
                        RGBColor hit_color = object->getColor(x, y);
                        double alpha = smoothstep(0, eps, -distance);
                        double Dedge = Dsmoothstep(0, eps, -distance);
//                        std::cout << distance << " " << Dedge << std::endl;
                        color = MixColors(hit_color, color, alpha);

                        for (auto& a: alphas) {
                            a *= (1 - alpha);
                        }
                        alphas.push_back(alpha);
                        Dedges.push_back(Dedge);
                        Dparams.push_back(object->Dparam(x, y));
                        Dcolors.push_back(object->Dcolor(x, y));
                    }

                    /// END SDF

                    // normalize alphas by number of samples
                    for (double& alpha : alphas) {
                        alpha /= num_samples;
                    }


                    // Calculate all the derivatives if reference is provided
                    if (reference.size()) {
                        auto Dmse = RGBColor{
                                color.r - reference(i, j, 0),
                                color.g - reference(i, j, 1),
                                color.b - reference(i, j, 2),
                        };

                        for (int i = 0; i < alphas.size(); ++i) {
                            auto alpha = alphas[i];
                            auto Dedge = Dedges[i];
                            auto Dparam = Dparams[i];
                            auto Dcolor = Dcolors[i];

                            for (auto& param : Dparam) {
                                param *= -Dedge * alpha * (Dmse.r + Dmse.g + Dmse.b);
                            }
                            /// rgb
                            for (int j = 0; j < Dcolor.size(); j += 3) {
                                Dcolor[j + 0] *= alpha * Dmse.r;
                                Dcolor[j + 1] *= alpha * Dmse.g;
                                Dcolor[j + 2] *= alpha * Dmse.b;
                            }

                            if (i == 0) {
                                mesh_.accumulateGrad(Dparam, Dcolor);
                            } else {
                                objects_[i - 1]->accumulateGrad(Dparam, Dcolor);
                            }
                        }

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
        std::cout << "YO" << std::endl;

        // fill retval
        {

        }
    }
};

#endif //SDF_SCENE_H
