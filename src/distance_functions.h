#ifndef SDF_DISTANCE_FUNCTIONS_H
#define SDF_DISTANCE_FUNCTIONS_H

#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <algorithm>

#include "color.h"
#include "geometry.h"

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
    virtual std::vector<double> RandomEdgePoints(std::mt19937 &rng, Vec2d& p, Vec2d &p_in, Vec2d &p_out) {
        return {};
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

    std::vector<double> RandomEdgePoints(std::mt19937 &rng, Vec2d& p, Vec2d &p_in, Vec2d &p_out) {
        std::uniform_real_distribution<double> th(-M_PI, M_PI);
        double theta = th(rng);
        double unit_x = sin(theta);
        double unit_y = cos(theta);

        Vec2d n(unit_x, unit_y);
        p = Vec2d(x_ + radius_ * unit_x, y_ + radius_ * unit_y);
        p_in = p - n * 1e-3;
        p_out = p + n * 1e-3;

        return {n.x, n.y, n.norm()};
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

//class Segment: public SDF {
//    double a_x_, a_y_, b_x_, b_y_;
//    RGBColor color_;
//public:
//    Segment(double a_x, double a_y, double b_x, double b_y, RGBColor color):
//        a_x_(a_x),
//        a_y_(a_y),
//        b_x_(b_x),
//        b_y_(b_y),
//        color_(color)
//    {}
//
//    double distance(double x, double y) override {
//        double dx = x - a_x_;
//        double dy = y - a_y_;
//        double bax = b_x_ - a_x_;
//        double bay = b_y_ - a_y_;
//
//        double h = std::clamp((dx * bax + dy * bay) / (bax * bax + bay * bay), 0.0, 1.0);
//        return std::sqrt((dx - bax * h) * (dx - bax * h) + (dy - bay * h) * (dy - bay * h));
//    }
//
//    RGBColor getColor(double x, double y) override {
//        return color_;
//    }
//
//};
//
//class AxisAlignedEquilateralTriangle: public SDF {
//    double x_, y_;
//    double radius_;
//    Color color_;
//public:
//    AxisAlignedEquilateralTriangle(double x, double y, double radius, Color color):
//        x_(x),
//        y_(y),
//        radius_(radius * 2 / std::sqrt(3)),
//        color_(color)
//    {}
//
//    double distance(double x, double y) override {
//        double k = std::sqrt(3.0);
//
//        double dx = std::abs(x - x_) - radius_;
//        double dy = y_ - y + radius_ / k + radius_ / std::sqrt(3) / 2;
//
//        if (dx + k * dy > 0.0) {
//            double tmp = dx;
//            dx = (dx - k * dy) / 2.0;
//            dy = (-k * tmp - dy) / 2.0;
//        }
//        dx -= std::clamp(dx, -2.0 * radius_, 0.0);
//
//        return (dy > 0.0 ? -1.0 : 1.0) * std::sqrt(dx * dx + dy * dy);
//    }
//
//    RGBColor getColor(double x, double y) override {
//        return color_.getColor(distance(x, y), y - y_ - radius_);
//    }
//};

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

//double sminCubic(double a, double b, double k)
//{
//    double h = std::max( k-abs(a-b), 0.0 )/k;
//    double m = h*h*h*0.5;
//    double s = m*k*(1.0/3.0);
//    return (a<b) ? a-s : b-s;
//}
//
//double sminCubicCol(double a, double b, double k)
//{
//    double h = std::max( k-abs(a-b), 0.0 )/k;
//    double m = h*h*h*0.5;
//    double s = m*k*(1.0/3.0);
//    return (a<b) ? m : 1-m;
//}
//
//class Intersection: public SDF {
//    std::shared_ptr<SDF> first_;
//    std::shared_ptr<SDF> second_;
//    bool smooth_;
//    double smoothness_;
//public:
//    Intersection(std::shared_ptr<SDF>&& first, std::shared_ptr<SDF>&& second,
//                 bool smooth=false, double smoothness=0.125):
//         first_(std::move(first)),
//         second_(std::move(second)),
//         smooth_(smooth),
//         smoothness_(smoothness)
//    {}
//
//    double distance(double x, double y) override {
//        if (smooth_) {
//            return sminCubic(first_->distance(x, y), second_->distance(x, y), smoothness_);
//        } else {
//            return std::min(first_->distance(x, y), second_->distance(x, y));
//        }
//    }
//
//    RGBColor getColor(double x, double y) override {
//        double first_dist = first_->distance(x, y);
//        double second_dist = second_->distance(x, y);
//        if (smooth_) {
//            double blend = sminCubicCol(second_dist, first_dist, smoothness_);
//            auto fcol = first_->getColor(x, y);
//            auto scol = second_->getColor(x, y);
//            return MixColors(fcol, scol, blend);
//        } else {
//            if (first_dist < second_dist) {
//                return first_->getColor(x, y);
//            } else {
//                return second_->getColor(x, y);
//            }
//        }
//    }
//};
//
//class Overlay: public SDF {
//    std::shared_ptr<SDF> top_;
//    std::shared_ptr<SDF> bottom_;
//    double alpha_;
//public:
//    Overlay(std::shared_ptr<SDF>&& top, std::shared_ptr<SDF>&& bottom, double alpha=0.5):
//        top_(std::move(top)),
//        bottom_(std::move(bottom)),
//        alpha_(alpha)
//    {}
//
//    double distance(double x, double y) override {
//        return std::min(top_->distance(x, y), bottom_->distance(x, y));
//    }
//
//    RGBColor getColor(double x, double y) override {
//        double top_dist = top_->distance(x, y);
//        double bottom_dist = bottom_->distance(x, y);
//
//        RGBColor top_color = top_->getColor(x, y);
//        RGBColor bottom_color = bottom_->getColor(x, y);
//
//        // TODO: move to get color definition
//        double eps = 2e-3;
//        if (top_dist < eps && bottom_dist < eps) {
//            return MixColors(top_color, bottom_color, alpha_);
//        } else if (bottom_dist < eps) {
//            return bottom_color;
//        } else {
//            return top_color;
//        }
//    }
//};

#endif //SDF_DISTANCE_FUNCTIONS_H
