#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>

int main(int argc, char** argv) {
    
    if (argc != 2) {
        std::cout << "Usage: ./demo_pointCloudVisualizer <ply_file>" << std::endl;
        return 0;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud (new pcl::PointCloud<pcl::PointXYZ>);

    pcl::PLYReader reader;

    int status = reader.read(argv[1], *point_cloud);
    if (status < 0) {
        std::cerr << "Failed to read point cloud" << std::endl;
    }

    pcl::visualization::PCLVisualizer viewer("Point cloud visualization");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_color_handler (point_cloud, 128, 128, 255);
    viewer.addPointCloud<pcl::PointXYZ>(point_cloud, cloud_color_handler, "cloud");
    viewer.addCoordinateSystem(2.0);

    while (!viewer.wasStopped ()) {
        viewer.spinOnce ();
    }

    return 0;
}