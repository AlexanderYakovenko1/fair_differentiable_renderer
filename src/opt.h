#ifndef OPT_H
#define OPT_H


#include <cmath>
#include <vector>


class Adam {
    double lr_;
    double beta1_, beta2_;
    double eps_;

    std::vector<double> first_;
    std::vector<double> second_;

public:
    Adam(const std::vector<double>& params, double lr, double beta1=0.9, double beta2=0.999, double eps=1e-8)
     : lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps) {
        first_.resize(params.size(), 0.);
        second_.resize(params.size(), 0.);
    }

    void step(std::vector<double>& params, const std::vector<double>& grad, int iter) {

        for (int i = 0; i < params.size(); ++i) {
            // no bias correction
            first_[i] = beta1_ * first_[i] + (1 - beta1_) * grad[i];
            second_[i] = beta2_ * second_[i] + (1 - beta2_) * grad[i] * grad[i];

            auto mu = first_[i];// / (1 - pow(beta1_, iter));
            auto sq = second_[i];// / (1 - pow(beta2_, iter));

            // inplace update
            params[i] -= lr_ * mu / (sqrt(sq) + eps_);
        }
    }
};

#endif //OPT_H
