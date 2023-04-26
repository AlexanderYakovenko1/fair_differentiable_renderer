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

    RGBColor sample(double x, double y, double eps=2e-3) {
        RGBColor color = background_;

        /// MESH
        if (mesh_.distance(x, y) < eps) {
            // as per task description has a_i = 1
            // therefore overrides any other color in pixel
            color = mesh_.getColor(x, y);
        }
        /// END MESH


        /// SDF
        for (const auto &object: objects_) {
            double distance = object->distance(x, y);
            RGBColor hit_color = object->getColor(x, y);
//            double alpha = smoothstep(0, eps, -distance);
            double alpha = (distance < eps);
            color = MixColors(hit_color, color, alpha);
        }
        /// END SDF

        return color;
    }
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
    std::vector<std::pair<std::vector<double>,std::vector<double>>> RenderToImage(Image<pixel_type>& image, std::mt19937& rng, int num_samples=10, double eps=1e-3,
                                                   const Image<double>& reference = Image<double>()) {
        // for each obect in the scene stores geometrical and color differentials
        std::vector<std::pair<std::vector<double>,std::vector<double>>> Dscene;

        std::uniform_real_distribution<double> dist(-0.5, 0.5);

        // zero the gradients
        mesh_.zeroGrad();
        for (const auto &object: objects_) {
            object->zeroGrad();
        }

        for (int i = 0; i < image.height(); ++i) {
            for (int j = 0; j < image.width(); ++j) {
                double y_ = y_min_ + double(i) / image.height() * (y_max_ - y_min_);
                double x_ = x_min_ + double(j) / image.width() * (x_max_ - x_min_);

                std::vector<RGBColor> colors;
                std::vector<double> weights;
                for (int trial = 0; trial < num_samples; ++trial) {
                    double x = x_ + dist(rng) / image.width();
                    double y = y_ + dist(rng) / image.height();

                    std::vector<double> alphas;  // alpha mixing coefficients of final color in sample
                    std::vector<std::vector<double>> Dparams;
                    std::vector<std::vector<double>> Dcolors;
                    std::vector<double> Dedges;
                    RGBColor color = background_;

                    /// MESH

                    Vec2d mesh_in, mesh_out;
                    bool mesh_edge;
                    if (mesh_.distance(x, y) < eps) {
                        // as per task description has a_i = 1
                        // therefore overrides any other color in pixel
                        color = mesh_.getColor(x, y);
                        alphas.push_back(1);
                    } else {
                        alphas.push_back(0);
                    }
                    Dedges.push_back(-1); // step func, edge sampling goes here i guess
                    Dparams.push_back(mesh_.Dmesh(x, y, mesh_in, mesh_out, mesh_edge));
                    Dcolors.push_back(mesh_.Dcolor(x, y));

                    RGBColor color_in{}, color_out{};
                    if (mesh_edge) {
                        color_in = sample(mesh_in.x, mesh_in.y, eps);
                        color_out = sample(mesh_out.x, mesh_out.y, eps);
                    }
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

                            if (i != 0) {
                                // SDF branch
                                for (auto &param: Dparam) {
                                    param *= -Dedge * alpha * (Dmse.r + Dmse.g + Dmse.b);
                                }
                            } else {
                                // Mesh branch
                                RGBColor in_out = {
                                        color_in.r - color_out.r,
                                        color_in.g - color_out.g,
                                        color_in.b - color_out.b,
                                };
                                for (auto &param: Dparam) {
                                    param *= alpha * (
                                            in_out.r * Dmse.r +
                                            in_out.g * Dmse.g +
                                            in_out.b * Dmse.b);
                                }
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
//        std::cout << "YO" << std::endl;

        // fill retval
        {
            Dscene.push_back(mesh_.grad());
            for (auto& object : objects_) {
                Dscene.push_back(object->grad());
            }
        }

        return Dscene;
    }

    std::vector<std::pair<std::vector<double>,std::vector<double>>> EdgeSampling(Image<uint8_t>& image, std::mt19937& rng, const Image<double>& reference, int num_samples=10000, double eps=1e-3) {

        // TODO: add flag
        // zero the gradients
        mesh_.zeroGrad();
        for (const auto &object: objects_) {
            object->zeroGrad();
        }

        std::vector<std::pair<std::vector<double>,std::vector<double>>> Dscene;

        for (int obj_idx = 0; obj_idx < objects_.size() + 1; ++ obj_idx) {
            SDF* obj;
            if (obj_idx == 0) {
                obj = &mesh_;
            } else {
                obj = objects_[obj_idx - 1].get();
            }

            auto Dcolor = obj->Dcolor(0.0, 0.0); // placeholder for ease of use
            for (auto &col: Dcolor) {
                col *= 0;
            }
            if (Dcolor.size() == 0) {
                continue;
            }
            for (int i = 0; i < num_samples; ++i) {
                Vec2d p, p_in, p_out;
                auto Dparam = obj->RandomEdgePoints(rng, p, p_in, p_out);

                int xi = round((p.x - x_min_) / (x_max_ - x_min_) * reference.width());
                int yi = round((p.y - y_min_) / (y_max_ - y_min_) * reference.height());

                if (xi < 0 || yi < 0 || xi > reference.width() || yi > reference.height()) {
                    continue;
                }

                auto color = sample(p.x, p.y, 0);
                auto color_in = sample(p_in.x, p_in.y, 0);
                auto color_out = sample(p_out.x, p_out.y, 0);

//                RGBColor ref_color = {
//                        reference(yi, xi, 0) != 0 ? color_in.r : 0.0,
//                        reference(yi, xi, 1) != 0 ? color_in.g : 0.0,
//                        reference(yi, xi, 2) != 0 ? color_in.b : 0.0,
//                };
                RGBColor ref_color = {
                        reference(yi, xi, 0),
                        reference(yi, xi, 1),
                        reference(yi, xi, 2),
                };

                auto adj = (color_in.r - color_out.r) * (color.r - ref_color.r) +
                           (color_in.g - color_out.g) * (color.g - ref_color.g) +
                           (color_in.b - color_out.b) * (color.b - ref_color.b);
                for (auto &param: Dparam) {
                    param *= adj;
                }

                obj->accumulateGrad(Dparam, Dcolor);

                image(yi, xi, 0) += 100;
            }
        }


        {
            Dscene.push_back(mesh_.grad());
            for (auto& object : objects_) {
                Dscene.push_back(object->grad());
            }
        }

        return Dscene;
    }

    std::vector<std::pair<std::vector<double>,std::vector<double>>> params() {
        std::vector<std::pair<std::vector<double>,std::vector<double>>> params;
        params.push_back(mesh_.params());
        for (auto& object : objects_) {
            params.push_back(object->params());
        }
        return params;
    }

    void updateScene(const std::vector<std::pair<std::vector<double>,std::vector<double>>>& new_params) {
        for (int i = 0; i < new_params.size(); ++i) {
            auto [params, color] = new_params[i];
            if (i == 0) {
                mesh_.updateParams(params, color);
            } else {
                objects_[i - 1]->updateParams(params, color);
            }
        }
    }
};

#endif //SDF_SCENE_H
