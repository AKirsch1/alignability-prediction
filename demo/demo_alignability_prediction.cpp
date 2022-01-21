#include <alignability_prediction.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>

// SIFT Keypoint parameters
const float min_scale = 0.01f;
const int n_octaves = 3;
const int n_scales_per_octave = 4;
const float min_contrast = 0.001f;

int main(int argc, char** argv) {
    
    if (argc != 4) {
        std::cout << "Usage: ./demo_alignabilityPrediction <source> <target> <top_percentage>" << std::endl;
        return 0;
    }

    AlignabilityPrediction<pcl::FPFHSignature33> comparison;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target (new pcl::PointCloud<pcl::PointXYZ>);

    pcl::PLYReader reader;

    int status = reader.read(argv[1], *cloud_source);
    if (status < 0) {
        std::cerr << "Failed to read source cloud" << std::endl;
    }

    status = reader.read(argv[2], *cloud_target);
    if (status < 0) {
        std::cerr << "Failed to read target cloud" << std::endl;
    }
    
    // Estimate cloud normals
    pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> ne;
    pcl::PointCloud<pcl::PointNormal>::Ptr src_normals_ptr (new pcl::PointCloud<pcl::PointNormal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_xyz (new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setInputCloud(cloud_source);
    ne.setSearchMethod(tree_xyz);
    ne.setKSearch(8);
    ne.compute(*src_normals_ptr);
    for(size_t i = 0;  i < src_normals_ptr->points.size(); ++i) {
        src_normals_ptr->points[i].x = cloud_source->points[i].x;
        src_normals_ptr->points[i].y = cloud_source->points[i].y;
        src_normals_ptr->points[i].z = cloud_source->points[i].z;
    }

    pcl::PointCloud<pcl::PointNormal>::Ptr tar_normals_ptr (new pcl::PointCloud<pcl::PointNormal>);
    ne.setInputCloud(cloud_target);
    ne.compute(*tar_normals_ptr);
    for(size_t i = 0;  i < tar_normals_ptr->points.size(); ++i) {
        tar_normals_ptr->points[i].x = cloud_target->points[i].x;
        tar_normals_ptr->points[i].y = cloud_target->points[i].y;
        tar_normals_ptr->points[i].z = cloud_target->points[i].z;
    }

    // Estimate the SIFT keypoints
    pcl::SIFTKeypoint<pcl::PointNormal, pcl::PointWithScale> sift;
    pcl::PointCloud<pcl::PointWithScale>::Ptr src_keypoints_ptr (new pcl::PointCloud<pcl::PointWithScale>);
    pcl::PointCloud<pcl::PointWithScale>& src_keypoints = *src_keypoints_ptr;
    pcl::search::KdTree<pcl::PointNormal>::Ptr tree_normal(new pcl::search::KdTree<pcl::PointNormal> ());
    sift.setSearchMethod(tree_normal);
    sift.setScales(min_scale, n_octaves, n_scales_per_octave);
    sift.setMinimumContrast(min_contrast);
    sift.setInputCloud(src_normals_ptr);
    sift.compute(src_keypoints);
        
    pcl::PointCloud<pcl::PointWithScale>::Ptr tar_keypoints_ptr (new pcl::PointCloud<pcl::PointWithScale>);
    pcl::PointCloud<pcl::PointWithScale>& tar_keypoints = *tar_keypoints_ptr;
    sift.setInputCloud(tar_normals_ptr);
    sift.compute(tar_keypoints);

    // Extract FPFH features from SIFT keypoints
    pcl::PointCloud<pcl::PointXYZ>::Ptr src_keypoints_xyz (new pcl::PointCloud<pcl::PointXYZ>);                           
    pcl::copyPointCloud (src_keypoints, *src_keypoints_xyz);
    pcl::FPFHEstimation<pcl::PointXYZ, pcl::PointNormal, pcl::FPFHSignature33> fpfh;
    fpfh.setSearchSurface (cloud_source);
    fpfh.setInputCloud (src_keypoints_xyz);
    fpfh.setInputNormals (src_normals_ptr);
    fpfh.setSearchMethod (tree_xyz);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr src_features_ptr (new pcl::PointCloud<pcl::FPFHSignature33>());
    pcl::PointCloud<pcl::FPFHSignature33>& src_features = *src_features_ptr;
    fpfh.setRadiusSearch(0.25);
    fpfh.compute(src_features);

    pcl::PointCloud<pcl::PointXYZ>::Ptr tar_keypoints_xyz (new pcl::PointCloud<pcl::PointXYZ>);                           
    pcl::copyPointCloud (tar_keypoints, *tar_keypoints_xyz);
    fpfh.setSearchSurface (cloud_target);
    fpfh.setInputCloud (tar_keypoints_xyz);
    fpfh.setInputNormals (tar_normals_ptr);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr tar_features_ptr (new pcl::PointCloud<pcl::FPFHSignature33>());
    pcl::PointCloud<pcl::FPFHSignature33>& tar_features = *tar_features_ptr;
    fpfh.compute(tar_features);

    comparison.setSourceFeatures(src_features_ptr);
    comparison.setTargetFeatures(tar_features_ptr);
    comparison.setTopPercentage(std::stof(argv[3]));

    comparison.calculateDistances();

    std::cout << "Values: " << std::endl;
    std::cout << "MedianFeatureDistance: " << comparison.getMedianFeatureDistance() << std::endl;
    std::cout << "MeanFeatureDistance: " << comparison.getMeanFeatureDistance() << std::endl;
    std::cout << "Variance: " << comparison.getFeatureVariance() << std::endl;
    std::cout << "StandardDeviation: " << comparison.getStandardDeviation() << std::endl;
    std::cout << "MinValue: " << comparison.getMinValue() << std::endl;
    std::cout << "MaxValue: " << comparison.getMaxValue() << std::endl; 
    std::cout << "Classification Result: " << comparison.isRegistrationPossible() << std::endl;

}
