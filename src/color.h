#ifndef COLOR_H
#define COLOR_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <numeric>

#include "image.h"

struct RGBColor {
    double r, g, b;
};

RGBColor MixColors(RGBColor first, RGBColor second, double alpha) {
    return {
            ((first.r) * alpha + (second.r) * (1 - alpha)),
            ((first.g) * alpha + (second.g) * (1 - alpha)),
            ((first.b) * alpha + (second.b) * (1 - alpha))
    };
}

// Weighted sum of squared color values
// Used in color blending
RGBColor MixColors(const std::vector<RGBColor>& colors, const std::vector<double>& weights) {
    if (colors.size() != weights.size()) {
        throw std::invalid_argument("Colors and weights vectors should have the same size");
    }

    double sum = std::accumulate(weights.begin(), weights.end(), 0.);
    RGBColor tmp{0., 0., 0.};
    for (int i = 0; i < colors.size(); ++i) {
        tmp.r += colors[i].r * weights[i];
        tmp.g += colors[i].g * weights[i];
        tmp.b += colors[i].b * weights[i];
    }

    return {(tmp.r / sum), (tmp.g / sum), (tmp.b / sum)};
}

// Weighted sum of linear color values
// Used it interpolation of images
RGBColor InterpolateColors(const std::vector<RGBColor>& colors, const std::vector<double>& weights) {
    if (colors.size() != weights.size()) {
        throw std::invalid_argument("Colors and weights vectors should have the same size");
    }

    double sum = std::accumulate(weights.begin(), weights.end(), 0.);
    RGBColor tmp{0., 0., 0.};
    for (int i = 0; i < colors.size(); ++i) {
        tmp.r += colors[i].r * colors[i].r * weights[i];
        tmp.g += colors[i].g * colors[i].g * weights[i];
        tmp.b += colors[i].b * colors[i].b * weights[i];
    }

    return {sqrt(tmp.r / sum), sqrt(tmp.g / sum), sqrt(tmp.b / sum)};
}

class Color {
    RGBColor base_;
    Image<double> texture_;
    std::vector<double> accumulated_;

    bool is_textured_;

    RGBColor sampleTexture(size_t x, size_t y) {
        return RGBColor{
            texture_(x, y, 0),
            texture_(x, y, 1),
            texture_(x, y, 2),
        };
    }

public:
    Color(RGBColor base):
            base_(base),
            is_textured_(false)
    {
        accumulated_.resize(3, 0.);
    }

    Color(Image<double>& texture):
            texture_(texture),
            is_textured_(true)
    {
        if (texture_.channels() != 3) {
            throw std::invalid_argument("Image must have 3 color channels");
        }
        accumulated_.resize(texture.size(), 0.);
    }

    RGBColor getColor(double u=0.0, double v=0.0) {
        if (is_textured_) {
            double tmp;

            // u and v must be from 0 to 1
            // cyclic padding
            u = modf(u, &tmp);
            v = modf(v, &tmp);

            u = u > 0 ? u : 1 + u;
            v = v > 0 ? v : 1 + v;

            double x = u * (texture_.width() - 1);
            double y = v * (texture_.height() - 1);

//            std::cout << x << " " << y << std::endl;

            double interp_x = modf(x, &tmp);
            double interp_y = modf(y, &tmp);

            std::vector<RGBColor> colors = {
                    sampleTexture(floor(x), floor(y)),
                    sampleTexture(ceil(x), floor(y)),
                    sampleTexture(floor(x), ceil(y)),
                    sampleTexture(ceil(x), ceil(y))
            };
            std::vector<double> weights = {
                    (1 - interp_y) * (1 - interp_x),
                    (1 - interp_y) * (    interp_x),
                    (    interp_y) * (1 - interp_x),
                    (    interp_y) * (    interp_x)
            };

            return InterpolateColors(colors, weights);
        } else {
            return base_;
        }
    }

    std::vector<double> grad() {
        return accumulated_;
    }

    void zeroGrad() {
        std::fill(accumulated_.begin(), accumulated_.end(), 0.0);
    }

    void accumulateGrad(const std::vector<double>& values) {
        for (int i = 0; i < accumulated_.size(); ++i) {
            accumulated_[i] += values[i];
        }
    }

    std::vector<double> Dcolor(double x, double y) {
        if (is_textured_) {
            return {}; // TODO: finish textured derivative
            // should return zero lattice with bilinear interpolation weights at (x, y)
        } else {
            // RGBcolor derivatives w.r.t. to each color
            return {1., 1., 1.};
        }
    }

    std::vector<double> params() {
        if (is_textured_) {
            return {};
        } else {
            return {base_.r, base_.g, base_.b};
        }
    }

    void updateParams(const std::vector<double>& values) {
        if (is_textured_) {
            return;
        } else {
            base_.r = std::clamp(values[0], 0., 1.);
            base_.g = std::clamp(values[1], 0., 1.);
            base_.b = std::clamp(values[2], 0., 1.);
        }
    }
};

#endif //COLOR_H
