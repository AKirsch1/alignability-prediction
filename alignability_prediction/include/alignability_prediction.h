#pragma once

#include <cmath>
#include <pcl/pcl_base.h>
#include <pcl/console/print.h>
#include <pcl/search/kdtree.h>
#include <boost/shared_ptr.hpp>
#include <pcl/point_representation.h>

template<typename FeatureT>
class AlignabilityPrediction {

public:
    using FeatureCloud = pcl::PointCloud<FeatureT>;
    using FeatureCloudPtr = typename FeatureCloud::Ptr;
    using FeatureCloudConstPtr = typename FeatureCloud::ConstPtr;

    using FeatureKdTree = pcl::KdTreeFLANN<FeatureT>;
    using FeatureKdTreePtr = typename FeatureKdTree::Ptr;

    using PointRepresentation = pcl::PointRepresentation<FeatureT>;
    using PointRepresentationConstPtr = typename PointRepresentation::ConstPtr;

    using Ptr = boost::shared_ptr<AlignabilityPrediction<FeatureT>>;
    using ConstPtr = boost::shared_ptr<const AlignabilityPrediction<FeatureT>>;

    AlignabilityPrediction()
    : source_features_()
    , target_features_()
    , kd_tree(new pcl::KdTreeFLANN<FeatureT>)
    , point_representation_(new pcl::DefaultPointRepresentation<FeatureT>)
    , top_percentage(1.0)
    {}

    ~AlignabilityPrediction() {}

    void setSourceFeatures(FeatureCloudConstPtr source_features) {
        source_features_ = source_features;
    }

    void setTargetFeatures(FeatureCloudConstPtr target_features) {
        target_features_ = target_features;
        kd_tree->setInputCloud(target_features_);
    }

    void setTopPercentage(float percentage) {
        top_percentage = std::min(std::max(percentage, 0.0f), 1.0f);
    }

    void calculateDistances() {
        if (!source_features_ || !target_features_) {
            PCL_WARN("[AlignabilityPrediction::calculateFeatureDistance]"
                     " Source or target features not set.\n");
            return;
        }

        // Set and Reset variables
        distances.clear();
        int count = source_features_->size();
        std::vector<int> nn_indices(1);
        std::vector<float> nn_distances(1);
        int query_dim = point_representation_->getNumberOfDimensions();

        // For every source feature
        for (const FeatureT& feature : *source_features_) {
            if (isFinite(feature, query_dim)) {
                // Find its closest target feature
                kd_tree->nearestKSearch(feature, 1, nn_indices, nn_distances);
                distances.push_back(nn_distances[0]);
            }
        }

        // Sort distances so min and max can easily be returned
        sort(distances.begin(), distances.end());
    }

    double getMedianFeatureDistance() {
        int size = std::round(distances.size() * top_percentage);
        if (size == 0) {
            return 0.0;
        } else if ((size % 2) == 1) {
            return distances[size / 2+1];
        } else {
            return (distances[size / 2] + distances[(size / 2) + 1]) / 2;
        }
    }

    double getMeanFeatureDistance() {
        double distance = 0.0;
        for (int i = 0; i < std::round(distances.size() * top_percentage); i++) {
            distance += distances[i];
        }
        return distance / std::round(distances.size() * top_percentage);
    }

    double getFeatureVariance() {
        double meanDistance = getMeanFeatureDistance();
        double counter = 0.0;
        for (int i = 0; i < std::round(distances.size() * top_percentage); i++) {
            counter += std::pow(distances[i] - meanDistance, 2);
        }
        return counter / std::round(distances.size() * top_percentage);
    }

    double getStandardDeviation() {
        return std::sqrt(getFeatureVariance());
    }

    double getMinValue() {
        return distances[0];
    }

    double getMaxValue() {
        return distances[std::round(distances.size() * top_percentage) - 1];
    }

    bool isRegistrationPossible() {
        // Execute SVM classifier in Python
        std::stringstream ss;
        ss << "python3 ./scripts/alignability_prediction.py "
           << this->getMeanFeatureDistance() << " " << this->getMedianFeatureDistance()
           << " " << this->getStandardDeviation() << " " << this->getMinValue() << " "
           << this->getMaxValue() << " 2>&1";
        std::string cmd = ss.str();

        // Parse Python output
        FILE* pipe = popen(cmd.c_str(), "r");
        if (pipe) {
            int buffer_length = 128;
            char buffer[buffer_length];
            std::string result = "";
            try {
                while (fgets(buffer, buffer_length, pipe) != NULL) {
                    result += buffer;
                }
            } catch (...) {
                pclose(pipe);
                return false;
            }
            
            pclose(pipe);
            
            if (result[0] == '1') {
                return true;
            } else if (result[0] == '0') {
                return false;
            } else {
                std::cout << "Error: Failed to predict alignability" << std::endl << result << std::endl;
            }
        }

        return false;
    }

private:
    FeatureCloudConstPtr source_features_;
    FeatureCloudConstPtr target_features_;

    FeatureKdTreePtr kd_tree;

    std::vector<double> distances;
    float top_percentage;

    PointRepresentationConstPtr point_representation_;

    bool isFinite(const FeatureT& feature, int query_dim) {
        std::vector<double> query(query_dim);
        point_representation_->vectorize(feature, query);

        // Only check first value as either none or all values should be NaN
        if (!std::isfinite(query[0])) {
            return false;
        }
        
        return true;
    }

};
