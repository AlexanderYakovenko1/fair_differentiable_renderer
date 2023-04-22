#ifndef SDF_IMAGE_H
#define SDF_IMAGE_H

#include <cstdint>
#include <vector>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h"

template <typename pixel_type>
class Image {
private:
    size_t height_, width_, channels_;
    size_t size_;
    std::vector<pixel_type> pixel_data_;
public:
    Image() = default;

    Image(const std::string& filepath, pixel_type div = pixel_type(1), int chans=3) {
        int width, height, channels;
        uint8_t* data = stbi_load(filepath.c_str(), &width, &height, &channels, chans);
        if (data == nullptr) {
            std::cerr << "Image failed to load" << std::endl;
            exit(1);
        }
        height_ = height;
        width_ = width;
        channels_ = channels;
        size_ = height * width * channels;

        pixel_data_ = std::vector<pixel_type>(size_, pixel_type());

        for (size_t i = 0; i < size_; ++i) {
            pixel_data_[i] = static_cast<pixel_type>(data[i]) / div;
        }
    }

    Image(size_t height, size_t width, size_t channels):
            height_(height),
            width_(width),
            channels_(channels),
            size_(height_ * width_ * channels_),
            pixel_data_(size_, pixel_type()) {}

    pixel_type& operator()(size_t x, size_t y, size_t z) {
        size_t idx = x * width_ * channels_ + y * channels_ + z;

        return pixel_data_[idx];
    }

    const pixel_type& operator()(size_t x, size_t y, size_t z) const {
        size_t idx = x * width_ * channels_ + y * channels_ + z;

        return pixel_data_[idx];
    }

    size_t size() const {
        return size_;
    }

    size_t width() const {
        return width_;
    }

    size_t height() const {
        return height_;
    }

    size_t channels() const {
        return channels_;
    }

    std::vector<pixel_type> data() {
        return pixel_data_;
    }

    void updateImage(const std::vector<pixel_type>& pixel_values) {
        std::copy(pixel_values.begin(), pixel_values.end(), pixel_data_.begin());
    }

    friend void Save8bitRgbImage(const std::string&, const Image<uint8_t>&);
};

void Save8bitRgbImage(const std::string& path, const Image<uint8_t>& image) {
    stbi_write_png(path.c_str(), image.width_, image.height_, STBI_rgb, image.pixel_data_.data(), image.width_ * image.channels_);
}
#endif //SDF_IMAGE_H
