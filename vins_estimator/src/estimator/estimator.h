/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once
 
#include <thread>
#include <mutex>
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <ceres/ceres.h>
#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include "sophus/se3.hpp"

#include "parameters.h"
#include "feature_manager.h"
#include "../utility/utility.h"
#include "../utility/tic_toc.h"
#include "../initial/solve_5pts.h"
#include "../initial/initial_sfm.h"
#include "../initial/initial_alignment.h"
#include "../initial/initial_ex_rotation.h"
#include "../factor/imu_factor.h"
#include "../factor/pose_local_parameterization.h"
#include "../factor/marginalization_factor.h"
#include "../factor/projectionTwoFrameOneCamFactor.h"
#include "../factor/projectionTwoFrameTwoCamFactor.h"
#include "../factor/projectionOneFrameTwoCamFactor.h"
#include "../featureTracker/feature_tracker.h"

typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
//typedef Eigen::Matrix<double, 2, 2> Matrix2d;
//typedef Eigen::Matrix<double, 2, 1> Vector2d;

class Estimator
{
  public:
    Estimator();

    void setParameter();

    // interface
    void initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r);
    void inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity);
    void inputFeature(double t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &featureFrame);
    void inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());
    void processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
    void processImage(const cv::Mat &image, const double header);
    void processMeasurements();

    // internal
    void clearState();
    bool initialStructure();
    bool visualInitialAlign();
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
    void slideWindow();
    void slideWindowNew();
    void slideWindowOld();
    void optimization();
    void vector2double();
    void double2vector();
    bool failureDetection();
    bool getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector, 
                                              vector<pair<double, Eigen::Vector3d>> &gyrVector);
    void getPoseInWorldFrame(Eigen::Matrix4d &T);
    void getPoseInWorldFrame(int index, Eigen::Matrix4d &T);
    void predictPtsInNextFrame();
    void outliersRejection(set<int> &removeIndex);
    double reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                     Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj, 
                                     double depth, Vector3d &uvi, Vector3d &uvj);
    void updateLatestStates();
    void fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity);
    bool IMUAvailable(double t);
    void initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector);

    void DirectPoseEstimation(const cv::Mat &_img1, const cv::Mat &_img2, const vector<pair<int, Eigen::Matrix<double, 7, 1>>> &_points, Sophus::SE3<double> &_T21, 
				const double n_fx, const double n_fy, const double n_cx, const double n_cy, const int level);

    void precomputeReferencePatches(const cv::Mat &img_ref, const vector<pair<int, Eigen::Matrix<double, 7, 1>>> &_points, 
	                               const double n_fx, const double n_fy, const double n_cx, const double n_cy);

    double computeResiduals(
         const cv::Mat &_img1,
         const cv::Mat &_img2,
         const vector<pair<int, Eigen::Matrix<double, 7, 1>>> &_points,
         Sophus::SE3<double> &_T21, const double n_fx, const double n_fy, const double n_cx, const double n_cy, const int level);

    void DirectPoseEstimationMultiLayer(const cv::Mat &_img1, const cv::Mat &_img2, const vector<pair<int, Eigen::Matrix<double, 7, 1>>> &_points_frame_9, Sophus::SE3<double> &_T21);
    double GetPixelValue(const cv::Mat &img, double x, double y);

//void DirectPoseEstimation(const cv::Mat &_img1, const cv::Mat &_img2, const vector<pair<int, Eigen::Matrix<double, 7, 1>>> &_points_frame_9, Sophus::SE3<double> &_T21, 
				//const double n_fx, const double n_fy, const double n_cx, const double n_cy);


    Vector2d world2cam(Vector3d xyz);
    void cam2world(const Vector2d& p, Vector3d& P);
    void distortion(const Vector2d& p_u, Vector2d& d_u);
    bool direct_Align2D(const cv::Mat &img1, const cv::Mat &img2,const Vector2d &puv_ref, Vector2d &puv_cur);
    void get_image_Pyramid(const cv::Mat &img1, const cv::Mat &img2, vector<cv::Mat> &pyr1,  vector<cv::Mat> &pyr2);
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> findMatch_by_Direct(const cv::Mat &img1, const cv::Mat &img2, 
											const vector<pair<int, Eigen::Matrix<double, 7, 1>>> &points, Sophus::SE3<double> &T21, const double header);

    double distance(cv::Point2f &pt1, cv::Point2f &pt2);
    //void findMatch_by_Direct(const cv::Mat &img1, const cv::Mat &img2, const vector<pair<int, Eigen::Matrix<double, 7, 1>>> &points, Sophus::SE3<double> &T21, const double header);
    bool judge_keyframe(const vector<pair<int, Eigen::Matrix<double, 7, 1>>> &_points);
    void add_into_feature(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &points_frame_10);
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> goodFeaturesToTrack(const cv::Mat &image1, const cv::Mat &image2, const vector<pair<int, Eigen::Matrix<double, 7, 1>>> &points);
    bool inBorder(const cv::Mat &image2, cv::Point2f points);
    void reducePoint2f(vector<cv::Point2f> &v, vector<uchar> status);
    void reduceVector(vector<int> &v, vector<uchar> status);
    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };

    std::mutex mBuf;
    queue<pair<double, Eigen::Vector3d>> accBuf;
    queue<pair<double, Eigen::Vector3d>> gyrBuf;
    queue<pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1> > > > > > featureBuf;
    queue<pair<double,cv::Mat>> img_CV_buf;
    double prevTime, curTime;
    bool openExEstimation;

    std::thread trackThread;
    std::thread processThread;

    FeatureTracker featureTracker;

    SolverFlag solver_flag;
    MarginalizationFlag  marginalization_flag;
    Vector3d g;

    Matrix3d ric[2];
    Vector3d tic[2];

    Vector3d        Ps[(WINDOW_SIZE + 1)];
    Vector3d        Vs[(WINDOW_SIZE + 1)];
    Matrix3d        Rs[(WINDOW_SIZE + 1)];
    Vector3d        Bas[(WINDOW_SIZE + 1)];
    Vector3d        Bgs[(WINDOW_SIZE + 1)];
    double td;

    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
    double Headers[(WINDOW_SIZE + 1)];
    vector<cv::Mat> image_in_frame;
    
    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];
    Vector3d acc_0, gyr_0;

    vector<double> dt_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

    int frame_count;
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;
    int inputImageCnt;

    FeatureManager f_manager;
    MotionEstimator m_estimator;
    InitialEXRotation initial_ex_rotation;

    bool first_imu;
    bool is_valid, is_key;
    bool failure_occur;

    vector<Vector3d> point_cloud;
    vector<Vector3d> margin_cloud;
    vector<Vector3d> key_poses;
    double initial_timestamp;

   int num_optic;
   int num_direct;
   int num_img;
   double time_optic;
   double time_direct;
   double time_img;

    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    double para_Feature[NUM_OF_F][SIZE_FEATURE];
    double para_Ex_Pose[2][SIZE_POSE];
    double para_Retrive_Pose[SIZE_POSE];
    double para_Td[1][1];
    double para_Tr[1][1];

    int loop_window_index;

    MarginalizationInfo *last_marginalization_info;
    vector<double *> last_marginalization_parameter_blocks;

    map<double, ImageFrame> all_image_frame;
    IntegrationBase *tmp_pre_integration;

    Eigen::Vector3d initP;
    Eigen::Matrix3d initR;

    double latest_time;
    Eigen::Vector3d latest_P, latest_V, latest_Ba, latest_Bg, latest_acc_0, latest_gyr_0;
    Eigen::Quaterniond latest_Q;

    bool initFirstPoseFlag;
    Eigen::Matrix<double, Eigen::Dynamic, 16, Eigen::RowMajor> ref_patch_cache_;
    Eigen::Matrix<double, 6, Eigen::Dynamic , Eigen::ColMajor> jacobian_cache_;
    vector<bool> visible_fts_;

    Eigen::Matrix<double, 6, 6>  H_;      
    Eigen::Matrix<double, 6, 1>  Jres_;   

    bool have_ref_patch_cache_;
};

class Feature_in_Frame
{
 public:
	Feature_in_Frame(const Eigen::Matrix<double, 7, 1> &_point)
	{
		point.x() = _point(0);
		point.y() = _point(1);
		point.z() = _point(2);
		uv.x() = _point(3);
		uv.y() = _point(4);
		velocity.x() = _point(5); 
		velocity.y() = _point(6); 
	}
	Vector3d point;
	Vector2d uv;
	Vector2d velocity;
};

class JacobianAccumulator 
{	
public:
	JacobianAccumulator
	(
	const cv::Mat &_img1,
	const cv::Mat &_img2,
	//const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &_points,
	const vector<pair<int, Eigen::Matrix<double, 7, 1>>> &_points,
	Sophus::SE3<double> &_T21) :
	img1(_img1), img2(_img2), points(_points), T21(_T21) 
	{}
			
	// accumulate jacobians in a range
         void accumulate_jacobian(const double n_fx, const double n_fy, const double n_cx, const double n_cy);
	// get hessian matrix
	Matrix6d hessian() const { return H; }
		
	// get bias
	Vector6d bias() const { return b; }
		
	// get total cost
	double cost_func() const { return cost; }
		
	// reset h, b, cost to zero
	void reset() 
	{
		H = Matrix6d::Zero();
		b = Vector6d::Zero();
		cost = 0;
	}
		
//private:
	const cv::Mat &img1;
	const cv::Mat &img2;
	//const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &points;
	const vector<pair<int, Eigen::Matrix<double, 7, 1>>> &points;
	Sophus::SE3<double> &T21;
		
	//std::mutex hessian_mutex;
	Matrix6d H = Matrix6d::Zero();
	Vector6d b = Vector6d::Zero();
	double cost = 0;
};



