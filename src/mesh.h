#ifndef MESH_H
#define MESH_H

#include "distance_functions.h"
#include "geometry.h"

class Triangle: public SDF {
    Vec2d p0_, p1_, p2_;
    Color col_;

public:
    Triangle(Vec2d p0, Vec2d p1, Vec2d p2, Color col):
            p0_(p0), p1_(p1), p2_(p2),
            col_(col)
    {}

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
        double x_min = std::min(p0_.x, std::min(p1_.x, p2_.x));
        double x_max = std::max(p0_.x, std::max(p1_.x, p2_.x));
        double y_min = std::min(p0_.y, std::min(p1_.y, p2_.y));
        double y_max = std::max(p0_.y, std::max(p1_.y, p2_.y));
        return col_.getColor((x - x_min) / (x_max - x_min), (y - y_min) / (y_max - y_min));
    }
};

class TriangleMesh: public SDF {
    // for simplicity's sake just nest triangles
    std::vector<Triangle> triangles_;

public:
    TriangleMesh() = default;

    TriangleMesh(const std::vector<Vec2d>& verts, const std::vector<Vec3i>& idxs, const std::vector<Color>& colors) {
        if (idxs.size() != colors.size()) {
            throw std::invalid_argument("Colors and weights vectors should have the same size");
        }

        for (int i = 0; i < idxs.size(); ++i) {
            triangles_.emplace_back(verts[idxs[i].x], verts[idxs[i].y], verts[idxs[i].z], colors[i]);
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
};

#endif //MESH_H
