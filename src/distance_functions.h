#ifndef SDF_DISTANCE_FUNCTIONS_H
#define SDF_DISTANCE_FUNCTIONS_H

#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <algorithm>

#include "color.h"

class SDF {
public:
    SDF() = default;
    virtual double distance(double x, double y) = 0;
    virtual ~SDF() = default;
    virtual RGBColor getColor(double x, double y) = 0;
    // Make abstract
    // Returns gradient vector in the order of construction
    virtual std::vector<double> Dparam(double x, double y) {
        return {};
    }
    virtual std::vector<double> Dcolor(double x, double y) {
        return {};
    }
    virtual void zeroGrad() {
        return;
    }
    virtual std::pair<std::vector<double>, std::vector<double>> params() {
        return {};
    }
    virtual std::pair<std::vector<double>, std::vector<double>> grad() {
        return {};
    }
    virtual void accumulateGrad(const std::vector<double>& param, const std::vector<double>& color) {
        return;
    }
    virtual void updateParams(const std::vector<double> &param, const std::vector<double> &color) {
        return;
    }
};

class Circle: public SDF {
    double x_, y_;
    double radius_;
    Color color_;

    std::vector<double> accumulated_;
public:
    Circle(double x, double y, double radius, Color color): x_(x), y_(y), radius_(radius), color_(color) {
        accumulated_.resize(3, 0.);
    }
    double distance(double x, double y) override {
        return std::sqrt((x - x_) * (x - x_) + (y - y_) * (y - y_)) - radius_;
    }
    RGBColor getColor(double x, double y) override {
        return color_.getColor((x - x_ + radius_) / 2 / radius_, (y - y_ + radius_) / 2 / radius_);
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
        auto params = std::vector<double>{x_, y_, radius_};
        return {params, color_.params()};
    }

    void updateParams(const std::vector<double> &param, const std::vector<double> &color) override {
        x_ = param[0];
        y_ = param[1];
        radius_ = std::max(param[2], 0.);

        color_.updateParams(color);
    }

    std::vector<double> Dparam(double x, double y) override {
        double dx = -(x - x_) / std::sqrt((x - x_) * (x - x_) + (y - y_) * (y - y_));
        double dy = -(y - y_) / std::sqrt((x - x_) * (x - x_) + (y - y_) * (y - y_));
        double dradius = -1;
        std::vector<double> d = std::vector<double>{dx, dy, dradius};
        return d;
    }

    std::vector<double> Dcolor(double x, double y) override {
        return color_.Dcolor(x, y);
    }
};

class AxisAlignedRectangle: public SDF {
    double x_, y_;
    double width_, height_;
    Color color_;

    std::vector<double> accumulated_;
public:
    AxisAlignedRectangle(double x, double y, double width, double height, Color color):
        x_(x),
        y_(y),
        width_(width),
        height_(height),
        color_(color)
    {
        accumulated_.resize(4, 0.);
    }

    double distance(double x, double y) override {
        double dx = std::abs(x - x_) - width_;
        double dy = std::abs(y - y_) - height_;

        return std::sqrt(std::max(dx, 0.0) * std::max(dx, 0.0) + std::max(dy, 0.0) * std::max(dy, 0.0)) + std::min(std::max(dx, dy), 0.0);
    }

    RGBColor getColor(double x, double y) override {
        return color_.getColor(distance(x, y), y - y_ - height_);
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
        auto params = std::vector<double>{x_, y_, width_, height_};
        return {params, color_.params()};
    }

    void updateParams(const std::vector<double> &param, const std::vector<double> &color) override {
        x_ = param[0];
        y_ = param[1];
        width_ = std::max(param[2], 0.);
        height_ = std::max(param[3], 0.);

        color_.updateParams(color);
    }

    std::vector<double> Dparam(double x, double y) override {
        double x_hat = std::abs(x - x_) - width_;
        double y_hat = std::abs(y - y_) - height_;

        double distance = std::sqrt(std::max(x_hat, 0.0) * std::max(x_hat, 0.0) + std::max(y_hat, 0.0) * std::max(y_hat, 0.0)) + std::min(std::max(x_hat, y_hat), 0.0);

        double Dx = -0.5 / distance * (2 * x_hat * (x - x_ > 0 ? 1 : -1) * (x_hat > 0) + (x - x_ > 0 ? 1 : -1) * (y_hat < x_hat && x_hat < 0.));
        double Dy = -0.5 / distance * (2 * y_hat * (y - y_ > 0 ? 1 : -1) * (y_hat > 0) + (y - y_ > 0 ? 1 : -1) * (x_hat < y_hat && y_hat < 0.));

        double Dwidth = 0.5 / distance * (2 * x_hat * -1 * (x_hat > 0) + -1 * (y_hat < x_hat && x_hat < 0.));
        double Dheight = 0.5 / distance * (2 * y_hat * -1 * (y_hat > 0) + -1 * (x_hat < y_hat && y_hat < 0.));

        return {-Dx, -Dy, -Dwidth, -Dheight};
    }

    std::vector<double> Dcolor(double x, double y) override {
        return color_.Dcolor(x, y);
    }
};

class SDFImage: public SDF {
    double x_, y_;
    double scale_;
    Color color_;

    int width_, height_;
    int max_side_;
    Image<double> image_;

    std::vector<double> accumulated_;

    double getPixel(size_t i, size_t j) {
        if (i < height_ && j < width_) {
            return image_(i, j, 0);
        } else {
            return 0;
        }
    }
public:
    SDFImage(const std::string& filepath, double x, double y, double scale, Color color): x_(x), y_(y), scale_(scale),
                                                                                          color_(color) {
        image_ = Image<double>(filepath, 255., 1);
        width_ = image_.width();
        height_ = image_.height();
        max_side_ = std::max(width_, height_);

        x_ = x_ - scale_ * width_ / max_side_ / 2;
        y_ = y_ - scale_ * height_ / max_side_ / 2;

        accumulated_.resize(height_ * width_, 0);
    }

    SDFImage(int height, int width, double x, double y, double scale, Color color)
     : height_(height), width_(width), x_(x), y_(y), scale_(scale), color_(color) {
        image_ = Image<double>(height_, width_, 1);

        for (int y = 0; y < height_; ++y) {
            for (int x = 0; x < width_; ++x) {
                image_(y, x, 0) = 0.5 + 1e-3;
            }
        }

        accumulated_.resize(height_ * width_, 0);

        max_side_ = std::max(width_, height_);

        x_ = x_ - scale_ * width_ / max_side_ / 2;
        y_ = y_ - scale_ * height_ / max_side_ / 2;
    }

    double distance(double x, double y) override {
        double ix = x - x_;
        double iy = y - y_;
        if (ix < 0 || iy < 0 || ix >= scale_ * width_ / max_side_ || iy >= scale_ * height_ / max_side_) {
            return 1.0;
        } else {
            // bilinear interpolation
            ix *= max_side_ / scale_;
            iy *= max_side_ / scale_;
            double interp_x = std::modf(ix, &ix);
            double interp_y = std::modf(iy, &iy);

            double pixels[4] = {
                    getPixel(static_cast<size_t>(iy    ), static_cast<size_t>(ix    )),
                    getPixel(static_cast<size_t>(iy    ), static_cast<size_t>(ix + 1)),
                    getPixel(static_cast<size_t>(iy + 1), static_cast<size_t>(ix    )),
                    getPixel(static_cast<size_t>(iy + 1), static_cast<size_t>(ix + 1))
            };
            double value = pixels[0] * (1 - interp_y) * (1 - interp_x) +
                           pixels[1] * (1 - interp_y) * (    interp_x) +
                           pixels[2] * (    interp_y) * (1 - interp_x) +
                           pixels[3] * (    interp_y) * (    interp_x);
            return 0.5 - value;
        }
    }

    RGBColor getColor(double x, double y) override {
        return color_.getColor(distance(x, y), y - y_ - scale_);
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
        // NOTE: size and scale are left out, only the pixel values
        auto params = image_.data();
        return {params, color_.params()};
    }

    void updateParams(const std::vector<double> &param, const std::vector<double> &color) override {
        image_.updateImage(param);
        color_.updateParams(color);
    }

    std::vector<double> Dparam(double x, double y) override {
        std::vector<double> d(width_ * height_, 0.);

        double ix = x - x_;
        double iy = y - y_;
        if (ix < 0 || iy < 0 || ix >= scale_ * width_ / max_side_ || iy >= scale_ * height_ / max_side_) {
            return d;
        } else {
            // bilinear interpolation
            ix *= max_side_ / scale_;
            iy *= max_side_ / scale_;
            double interp_x = std::modf(ix, &ix);
            double interp_y = std::modf(iy, &iy);

            if (iy + 1 < height_ && ix + 1 < width_) {
                d[static_cast<size_t>(iy    ) * width_ + static_cast<size_t>(ix    )] = -(1 - interp_y) * (1 - interp_x);
                d[static_cast<size_t>(iy    ) * width_ + static_cast<size_t>(ix + 1)] = -(1 - interp_y) * (    interp_x);
                d[static_cast<size_t>(iy + 1) * width_ + static_cast<size_t>(ix    )] = -(    interp_y) * (1 - interp_x);
                d[static_cast<size_t>(iy + 1) * width_ + static_cast<size_t>(ix + 1)] = -(    interp_y) * (    interp_x);
            }
        }
        return d;
    }

    std::vector<double> Dcolor(double x, double y) override {
        return color_.Dcolor(x, y);
    }
};

#endif //SDF_DISTANCE_FUNCTIONS_H
