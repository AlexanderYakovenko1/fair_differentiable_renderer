#ifndef MESH_H
#define MESH_H

#include "distance_functions.h"
#include "geometry.h"

class Triangle: public SDF {
    Vec2d p0_, p1_, p2_;
    Color color_;

    std::vector<double> accumulated_;
public:
    Triangle(Vec2d p0, Vec2d p1, Vec2d p2, Color col):
            p0_(p0), p1_(p1), p2_(p2),
            color_(col)
    {
        accumulated_.resize(6, 0.);
    }

    double distance(double x, double y) override {
        Vec2d pos(x, y);
        Vec2d n0 = (p1_ - p0_).transpose();
        Vec2d n1 = (p2_ - p1_).transpose();
        Vec2d n2 = (p0_ - p2_).transpose();

        auto in0 = (pos - p0_) * n0 > 0;
        auto in1 = (pos - p1_) * n1 > 0;
        auto in2 = (pos - p2_) * n2 > 0;

        if ((in0 && in1 && in2) || (!in0 && !in1 && !in2)) {
            return 0;
        } else {
            return 1;
        }
    }

    RGBColor getColor(double x, double y) override {
        Vec2d v0 = p1_ - p0_;
        Vec2d v1 = p2_ - p0_;
        Vec2d v2 = Vec2d(x, y) - p0_;

        double d00 = v0 * v0;
        double d01 = v0 * v1;
        double d11 = v1 * v1;
        double d20 = v2 * v0;
        double d21 = v2 * v1;
        double denom = d00 * d11 - d01 * d01;
        double v = (d11 * d20 - d01 * d21) / denom;
        double w = (d00 * d21 - d01 * d20) / denom;
        double u = 1.0f - v - w;

        return color_.getColor(w, v);
    }

    std::pair<std::vector<double>, std::vector<double>> grad() override {
        return {accumulated_, color_.grad()};
    }

    void zeroGrad() override {
        std::fill(accumulated_.begin(), accumulated_.end(), 0.0);
        color_.zeroGrad();
    }

    void accumulateGrad(const std::vector<double> &param, const std::vector<double> &color) override {
        for (int i = 0; i < accumulated_.size(); ++i) {
            accumulated_[i] += param[i];
        }
        color_.accumulateGrad(color);
    }

    std::pair<std::vector<double>, std::vector<double>> params() override {
        auto params = std::vector<double>{p0_.x, p0_.y, p1_.x, p1_.y, p2_.x, p2_.y};
        return {params, color_.params()};
    }

    void updateParams(const std::vector<double> &param, const std::vector<double> &color) override {
        p0_.x = param[0];
        p0_.y = param[1];
        p1_.x = param[2];
        p1_.y = param[3];
        p2_.x = param[4];
        p2_.y = param[5];

        color_.updateParams(color);
    }

    std::vector<double> Dparam(double x, double y) override {
        // edge sampling(?)
        return {0., 0., 0., 0., 0., 0.};
    }

    std::vector<double> Dcolor(double x, double y) override {
        Vec2d v0 = p1_ - p0_;
        Vec2d v1 = p2_ - p0_;
        Vec2d v2 = Vec2d(x, y) - p0_;

        double d00 = v0 * v0;
        double d01 = v0 * v1;
        double d11 = v1 * v1;
        double d20 = v2 * v0;
        double d21 = v2 * v1;
        double denom = d00 * d11 - d01 * d01;
        double v = (d11 * d20 - d01 * d21) / denom;
        double w = (d00 * d21 - d01 * d20) / denom;
        double u = 1.0f - v - w;

        return color_.Dcolor(w, v);
    }
};

class TriangleMesh: public SDF {
    // for simplicity's sake just nest triangles
    std::vector<Triangle> triangles_;
    // sizes of Dparam and Dcolor
    std::vector<std::pair<int, int>> param_dims;
public:
    TriangleMesh() = default;

    TriangleMesh(const std::vector<Vec2d>& verts, const std::vector<Vec3i>& idxs, const std::vector<Color>& colors) {
        if (idxs.size() != colors.size()) {
            throw std::invalid_argument("Colors and weights vectors should have the same size");
        }

        for (int i = 0; i < idxs.size(); ++i) {
            triangles_.emplace_back(verts[idxs[i].x], verts[idxs[i].y], verts[idxs[i].z], colors[i]);
            auto [geom, color] = triangles_.back().grad();
            param_dims.emplace_back(geom.size(), color.size());
        }
    }

    double distance(double x, double y) override {
        for (auto& tr : triangles_) {
            if (tr.distance(x, y) == 0) {
                return 0;
            }
        }
        return 1;
    }

    RGBColor getColor(double x, double y) override {
        for (auto& tr : triangles_) {
            if (tr.distance(x, y) == 0) {
                return tr.getColor(x, y);
            }
        }
        return {};
    }

    std::pair<std::vector<double>, std::vector<double>> grad() override {
        std::pair<std::vector<double>, std::vector<double>> g{{}, {}};
        for (auto& tr : triangles_) {
            auto [geom, color] = tr.grad();
            g.first.insert(g.first.end(), geom.begin(), geom.end());
            g.second.insert(g.second.end(), color.begin(), color.end());
        }
        return g;
    }

    void zeroGrad() override {
        for (auto& tr : triangles_) {
            tr.zeroGrad();
        }
    }

    void accumulateGrad(const std::vector<double> &param, const std::vector<double> &color) override {
        auto param_begin = param.begin();
        auto color_begin = color.begin();
        for (int i = 0; i < param_dims.size(); ++i) {
            auto [param_size, color_size] = param_dims[i];
            triangles_[i].accumulateGrad(
                    std::vector<double>(param_begin, param_begin + param_size),
                    std::vector<double>(color_begin, color_begin + color_size));
            param_begin = std::next(param_begin, param_size);
            color_begin = std::next(color_begin, color_size);
        }
    }

    std::pair<std::vector<double>, std::vector<double>> params() override {
        std::pair<std::vector<double>, std::vector<double>> p{std::vector<double>(), std::vector<double>()};
        for (auto& tr : triangles_) {
            auto [geom, color] = tr.params();
            p.first.insert(p.first.end(), geom.begin(), geom.end());
            p.second.insert(p.second.end(), color.begin(), color.end());
        }
        return p;
    }

    void updateParams(const std::vector<double> &param, const std::vector<double> &color) override {
        auto param_begin = param.begin();
        auto color_begin = color.begin();
        for (int i = 0; i < param_dims.size(); ++i) {
            auto [param_size, color_size] = param_dims[i];
            triangles_[i].updateParams(
                    std::vector<double>(param_begin, param_begin + param_size),
                    std::vector<double>(color_begin, color_begin + color_size));
            param_begin = std::next(param_begin, param_size);
            color_begin = std::next(color_begin, color_size);
        }
    }

    std::vector<double> Dparam(double x, double y) override {
        std::vector<double> d;
        for (auto& tr : triangles_) {
            auto D = tr.Dparam(x, y);
            d.insert(d.end(), D.begin(), D.end());
        }

        return d;
    }

    std::vector<double> Dcolor(double x, double y) override {
        std::vector<double> d;
        for (auto& tr : triangles_) {
            auto D = tr.Dcolor(x, y);

            if (tr.distance(x, y) != 0) {
                for (auto& val: D) {
                    val = 0.0;
                }
            }

            d.insert(d.end(), D.begin(), D.end());
        }

        return d;
    }
};

#endif //MESH_H
