#ifndef MESH_H
#define MESH_H

#include "distance_functions.h"
#include "geometry.h"

class Triangle: public SDF {
    Vec2d p0_, p1_, p2_;
    Color color_;

    std::vector<double> accumulated_;

    std::pair<double, Vec2d> closestPoint(Vec2d p, Vec2d a, Vec2d b) {
        auto ab = b - a;
        auto ap = p - a;

        auto t = ab * ap / (ab * ab);
        auto closest = a + b * t;
        return {t, closest};
    }
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

    std::vector<double> Dmesh(double x, double y, Vec2d& p_in, Vec2d& p_out, bool& found) {
        // edge sampling(?)
        std::vector<double> d(6, 0.);
        return d;

        // finds closest segment
        Vec2d p(x, y);

        int closest_idx = -1;
        double closest_t;
        Vec2d closest_point;
        Vec2d closest_normal;
        double closest_dist = std::numeric_limits<double>::max();
        int i = 0;
        for (auto [a, b] : {std::make_pair(p0_, p1_),
                            std::make_pair(p1_, p2_),
                            std::make_pair(p2_, p0_)}) {
            auto [t, closest] = closestPoint(p, a, b);

            auto dist = (p - closest).norm();
            if (0 < t && t < 1 && dist < closest_dist) {
                closest_idx = i;
                closest_t = t;
                closest_point = closest;
                closest_normal = -(b - a).normalize().transpose();

                closest_dist = dist;
            }
            ++i;
        }

        double THR = 1e-2;
        if (closest_idx != -1 && closest_dist < THR) {
            // mirror p on closest edge to get inner or outer point of triangle
//            auto other = p + (closest_point - p) * 2;
//            auto normal = (closest_point - p).normalize();
//            if (distance(x, y) < 1e-3) {
//                normal = -normal;
//            }
            p_in = p - closest_normal * 1e-3;
            p_out = p + closest_normal * 1e-3;
            found = true;
//            std::cout << "idx: " << closest_idx << std::endl;
            if (closest_idx == 0) {
                d[0] = (1 - closest_t) * closest_normal.x;
                d[1] = (1 - closest_t) * closest_normal.y;
                d[2] = (    closest_t) * closest_normal.x;
                d[3] = (    closest_t) * closest_normal.y;
            } else if (closest_idx == 1) {
                d[2] = (1 - closest_t) * closest_normal.x;
                d[3] = (1 - closest_t) * closest_normal.y;
                d[4] = (    closest_t) * closest_normal.x;
                d[5] = (    closest_t) * closest_normal.y;
            } else {
                d[4] = (1 - closest_t) * closest_normal.x;
                d[5] = (1 - closest_t) * closest_normal.y;
                d[0] = (    closest_t) * closest_normal.x;
                d[1] = (    closest_t) * closest_normal.y;
            }
        }

        return d;
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

    std::vector<double> Dmesh(double x, double y, Vec2d& p_in, Vec2d& p_out, bool& found) {
        std::vector<double> d;
        for (auto& tr : triangles_) {
            auto D = tr.Dmesh(x, y, p_in, p_out, found);
            d.insert(d.end(), D.begin(), D.end());
            if (found) break;
        }
        auto pad = std::vector<double>(triangles_.size() * 6 - d.size(), 0.0);
        d.insert(d.end(), pad.begin(), pad.end());

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
