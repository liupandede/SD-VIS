/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "estimator.h"
#include "../utility/visualization.h"

Estimator::Estimator(): f_manager{Rs}
{
    ROS_INFO("init begins");
    clearState();
    prevTime = -1;
    curTime = 0;
    openExEstimation = 0;
    initP = Eigen::Vector3d(0, 0, 0);
    initR = Eigen::Matrix3d::Identity();
    inputImageCnt = 0;
    initFirstPoseFlag = false;
}

void Estimator::setParameter()
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
        cout << " exitrinsic cam " << i << endl  << ric[i] << endl << tic[i].transpose() << endl;
    }
    f_manager.setRic(ric);
    ProjectionTwoFrameOneCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTwoFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionOneFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD;
    g = G;
    cout << "set g " << g.transpose() << endl;
    featureTracker.readIntrinsicParameter(CAM_NAMES);

    std::cout << "MULTIPLE_THREAD is " << MULTIPLE_THREAD << '\n';
    if (MULTIPLE_THREAD)
    {
        processThread   = std::thread(&Estimator::processMeasurements, this);
    }
}

void Estimator::inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1)
{
    //inputImageCnt++;
    //map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    //TicToc featureTrackerTime;
    //if(_img1.empty())
        //featureFrame = featureTracker.trackImage(t, _img);
    //else
        //featureFrame = featureTracker.trackImage(t, _img, _img1);
    //printf("featureTracker time: %f\n", featureTrackerTime.toc());
    
    if(MULTIPLE_THREAD)  
    {     
        //if(inputImageCnt % 2 == 0)
        //{
            mBuf.lock();
            img_CV_buf.push(make_pair(t, _img));
            mBuf.unlock();
        //}
    }
    else
    {
        mBuf.lock();
        img_CV_buf.push(make_pair(t, _img));
        mBuf.unlock();
        TicToc processTime;
        processMeasurements();
        printf("process time: %f\n", processTime.toc());
    }
    
}

void Estimator::inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity)
{
    mBuf.lock();
    accBuf.push(make_pair(t, linearAcceleration));
    gyrBuf.push(make_pair(t, angularVelocity));
    //printf("input imu with time %f \n", t);
    mBuf.unlock();

    fastPredictIMU(t, linearAcceleration, angularVelocity);
    if (solver_flag == NON_LINEAR)
        pubLatestOdometry(latest_P, latest_Q, latest_V, t);
}

void Estimator::inputFeature(double t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &featureFrame)
{
    mBuf.lock();
    featureBuf.push(make_pair(t, featureFrame));
    mBuf.unlock();

    if(!MULTIPLE_THREAD)
        processMeasurements();
}


bool Estimator::getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector, 
                                vector<pair<double, Eigen::Vector3d>> &gyrVector)
{
    if(accBuf.empty())
    {
        printf("not receive imu\n");
        return false;
    }
    //printf("get imu from %f %f\n", t0, t1);
    //printf("imu fornt time %f   imu end time %f\n", accBuf.front().first, accBuf.back().first);
    if(t1 <= accBuf.back().first)
    {
        while (accBuf.front().first <= t0)
        {
            accBuf.pop();
            gyrBuf.pop();
        }
        while (accBuf.front().first < t1)
        {
            accVector.push_back(accBuf.front());
            accBuf.pop();
            gyrVector.push_back(gyrBuf.front());
            gyrBuf.pop();
        }
        accVector.push_back(accBuf.front());
        gyrVector.push_back(gyrBuf.front());
    }
    else
    {
        printf("wait for imu\n");
        return false;
    }
    return true;
}

bool Estimator::IMUAvailable(double t)
{
    if(!accBuf.empty() && t <= accBuf.back().first)
        return true;
    else
        return false;
}

void Estimator::processMeasurements()
{
    while (1)
    {
        //printf("process measurments\n");
        //pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1> > > > > feature;
	pair<double,cv::Mat> img_frame;
        vector<pair<double, Eigen::Vector3d>> accVector, gyrVector;
        if(!img_CV_buf.empty())
        {
            img_frame = img_CV_buf.front();
            curTime = img_frame.first + td;
            while(1)
            {
                if ((!USE_IMU  || IMUAvailable(img_frame.first + td)))
                    break;
                else
                {
                    printf("wait for imu ... \n");
                    if (! MULTIPLE_THREAD)
                        return;
                    std::chrono::milliseconds dura(5);
                    std::this_thread::sleep_for(dura);
                }
            }
            mBuf.lock();
            if(USE_IMU)
                getIMUInterval(prevTime, curTime, accVector, gyrVector);

            img_CV_buf.pop();
            mBuf.unlock();

            if(USE_IMU)
            {
                if(!initFirstPoseFlag)
                    initFirstIMUPose(accVector);
                for(size_t i = 0; i < accVector.size(); i++)
                {
                    double dt;
                    if(i == 0)
                        dt = accVector[i].first - prevTime;
                    else if (i == accVector.size() - 1)
                        dt = curTime - accVector[i - 1].first;
                    else
                        dt = accVector[i].first - accVector[i - 1].first;
                    processIMU(accVector[i].first, dt, accVector[i].second, gyrVector[i].second);
                }
            }

            processImage(img_frame.second, img_frame.first);
            prevTime = curTime;

            printStatistics(*this, 0);

            std_msgs::Header header;
            header.frame_id = "world";
            header.stamp = ros::Time(img_frame.first);

            pubOdometry(*this, header);
            pubKeyPoses(*this, header);
            pubCameraPose(*this, header);
            pubPointCloud(*this, header);
            pubKeyframe(*this);
            pubTF(*this, header);
        }

        if (! MULTIPLE_THREAD)
            break;

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}


void Estimator::initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector)
{
    printf("init first imu pose\n");
    initFirstPoseFlag = true;
    //return;
    Eigen::Vector3d averAcc(0, 0, 0);
    int n = (int)accVector.size();
    for(size_t i = 0; i < accVector.size(); i++)
    {
        averAcc = averAcc + accVector[i].second;
    }
    averAcc = averAcc / n;
    printf("averge acc %f %f %f\n", averAcc.x(), averAcc.y(), averAcc.z());
    Matrix3d R0 = Utility::g2R(averAcc);
    double yaw = Utility::R2ypr(R0).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    Rs[0] = R0;
    cout << "init R0 " << endl << Rs[0] << endl;
    //Vs[0] = Vector3d(5, 0, 0);
}

void Estimator::initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r)
{
    Ps[0] = p;
    Rs[0] = r;
    initP = p;
    initR = r;
}


void Estimator::clearState()
{
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
        {
            delete pre_integrations[i];
        }
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();

    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;
}

void Estimator::processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    if (frame_count != 0)
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        //if(solver_flag != NON_LINEAR)
            tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        int j = frame_count;         
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity; 
}

double Estimator::GetPixelValue(const cv::Mat &img, double x, double y) {
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols) x = img.cols - 1;
    if (y >= img.rows) y = img.rows - 1;
    uchar *data = &img.data[int(y) * img.step + int(x)];
    double xx = x - floor(x);
    double yy = y - floor(y);
    return double(
        (1 - xx) * (1 - yy) * data[0] +
        xx * (1 - yy) * data[1] +
        (1 - xx) * yy * data[img.step] +
        xx * yy * data[img.step + 1]
    );
}

/*void JacobianAccumulator::accumulate_jacobian(const double n_fx, const double n_fy, const double n_cx, const double n_cy)
{
    const int half_patch_size = 2;
    int cnt_good = 0;
    Matrix6d hessian = Matrix6d::Zero();
    Vector6d bias = Vector6d::Zero();
    double cost_tmp = 0;
	
    for (auto &id_pts : points)
    {
	Feature_in_Frame f_in_fra9(id_pts.second);
	const Vector3d xyz_ref = f_in_fra9.point;
	const double u_ref = f_in_fra9.uv(0);
        const double v_ref = f_in_fra9.uv(1);
	Vector3d xyz_cur = T21 * xyz_ref;
	if (xyz_cur[2] < 0)
	{   // depth invalid
            continue;
	}
	double dep_j = xyz_cur(2);
        const double u_cur = xyz_cur(0) / dep_j;
        const double v_cur = xyz_cur(1) / dep_j;
	double n_k1=-0.2917;
	double n_k2= 8.228e-02;
	double n_p1=5.333e-05;
	double n_p2=-1.578e-04;
	//double n_fx=4.616e+02;
	//double n_fy=4.603e+02;
	//double n_cx=3.630e+02;
	//double n_cy=2.481e+02;
	double mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u,add_distort_u,add_distort_v,u,v;
	mx2_u = u_cur * u_cur;
        my2_u = v_cur * v_cur;
        mxy_u = u_cur * v_cur;
	rho2_u = mx2_u + my2_u;
        rad_dist_u = n_k1 * rho2_u + n_k2 * rho2_u * rho2_u;
	add_distort_u=u_cur+u_cur * rad_dist_u + 2.0 * n_p1 * mxy_u + n_p2 * (rho2_u + 2.0 * mx2_u);
	add_distort_v=v_cur+v_cur * rad_dist_u + 2.0 * n_p2 * mxy_u + n_p1 * (rho2_u + 2.0 * my2_u);
	u=add_distort_u*n_fx+n_cx;
	v=add_distort_v*n_fy+n_cy;
	if (u < half_patch_size || u > img2.cols - half_patch_size || v < half_patch_size || v > img2.rows - half_patch_size)
		continue;
	double X = xyz_cur[0], Y = xyz_cur[1], Z = xyz_cur[2], Z_inv = 1.0 / Z, Z2_inv = Z_inv * Z_inv;
	cnt_good++;
	for (int x = -half_patch_size; x <= half_patch_size; x++)
		for (int y = -half_patch_size; y <= half_patch_size; y++) {
	
			double error = GetPixelValue(img1, u_ref + x, v_ref + y) - GetPixelValue(img2, u + x, v + y);
			Matrix<double, 2, 6> J_pixel_xi;
			Vector2d J_img_pixel;
	
			J_pixel_xi(0, 0) = n_fx * Z_inv;
			J_pixel_xi(0, 1) = 0;
			J_pixel_xi(0, 2) = -n_fx * X * Z2_inv;
			J_pixel_xi(0, 3) = -n_fx * X * Y * Z2_inv;
			J_pixel_xi(0, 4) = n_fx + n_fx * X * X * Z2_inv;
			J_pixel_xi(0, 5) = -n_fx * Y * Z_inv;
	
			J_pixel_xi(1, 0) = 0;
			J_pixel_xi(1, 1) = n_fy * Z_inv;
			J_pixel_xi(1, 2) = -n_fy * Y * Z2_inv;
			J_pixel_xi(1, 3) = -n_fy - n_fy * Y * Y * Z2_inv;
			J_pixel_xi(1, 4) = n_fy * X * Y * Z2_inv;
			J_pixel_xi(1, 5) = n_fy * X * Z_inv;
	
			J_img_pixel = Vector2d(
				0.5 * (GetPixelValue(img2, u + 1 + x, v + y) - GetPixelValue(img2, u - 1 + x, v + y)),
				0.5 * (GetPixelValue(img2, u + x, v + 1 + y) - GetPixelValue(img2, u + x, v - 1 + y))
			);
	
			// total jacobian
			Vector6d J = -1.0 * (J_img_pixel.transpose() * J_pixel_xi).transpose();
	
			hessian += J * J.transpose();
			bias += -error * J;
			cost_tmp += error * error;
		}
	}
	 if (cnt_good) 
	 {
            // set hessian, bias and cost
	    //unique_lock<mutex> lck(hessian_mutex);
	    //hessian_mutex.lock();
            H = hessian;
            b = bias;
            cost = cost_tmp / cnt_good;
	    //hessian_mutex.unlock();
   	 }
}

void DirectPoseEstimation(
    const cv::Mat &_img1,
    const cv::Mat &_img2,
    const vector<pair<int, Eigen::Matrix<double, 7, 1>>> &_points,
    Sophus::SE3<double> &_T21, const double n_fx, const double n_fy, const double n_cx, const double n_cy)
{
	const int iterations = 20;
	double cost = 0, lastCost = 0;
	JacobianAccumulator jaco_accu(_img1, _img2, _points, _T21);
	for (int iter = 0; iter < iterations; iter++)
	{
	    jaco_accu.reset();
	    jaco_accu.accumulate_jacobian(n_fx,n_fy,n_cx,n_cy);
	    Matrix6d H = jaco_accu.hessian();
            Vector6d b = jaco_accu.bias();

            // solve update and put it into estimation
            Vector6d update = H.ldlt().solve(b);
            jaco_accu.T21 = Sophus::SE3<double>::exp(update) * jaco_accu.T21;
            cost = jaco_accu.cost_func();

            if (std::isnan(update[0])) {
                // sometimes occurred when we have a black or white patch and H is irreversible
                //cout << "update is nan" << endl;
                break;
            }
            if (iter > 0 && cost > lastCost) {
                //cout << "cost increased: " << cost << ", " << lastCost << endl;
                break;
            }
            if (update.norm() < 1e-3) {
                //cout << "update is converge" << endl;
                break;
            }

            lastCost = cost;
            //cout << "iteration: " << iter << ", cost: " << cost << endl;
	}
	_T21 = jaco_accu.T21;
	//cout << "_T21 = \n" << _T21.matrix() << endl;

}*/

void Estimator::precomputeReferencePatches(const cv::Mat &img_ref, const vector<pair<int, Eigen::Matrix<double, 7, 1>>> &_points, 
	 const double n_fx, const double n_fy, const double n_cx, const double n_cy)
{
	const int border = 3;
	const int half_patch_size = 2;
	const int patch_area_ = 16;

  	int feature_counter = 0;
  	vector<bool>::iterator visiblity_it = visible_fts_.begin();

	for(auto it = _points.begin(), ite = _points.end(); it!=ite; ++it, ++feature_counter, ++visiblity_it)
       	{
		Feature_in_Frame f_in_fra9((*it).second);
		const Vector3d xyz_ref = f_in_fra9.point;
		const double u_ref = f_in_fra9.uv(0);
        	const double v_ref = f_in_fra9.uv(1);

		if(u_ref-border < 0 || v_ref-border < 0 || u_ref+border >= img_ref.cols || v_ref+border >= img_ref.rows)
      			continue;

    		*visiblity_it = true;

		double X = xyz_ref[0], Y = xyz_ref[1], Z = xyz_ref[2], Z_inv = 1.0 / Z, Z2_inv = Z_inv * Z_inv;
		Matrix<double, 2, 6> J_pixel_xi;
	
		J_pixel_xi(0, 0) = -n_fx * Z_inv;
		J_pixel_xi(0, 1) = 0;
		J_pixel_xi(0, 2) = n_fx * X * Z2_inv;
		J_pixel_xi(0, 3) = n_fx * X * Y * Z2_inv;
		J_pixel_xi(0, 4) = -(n_fx + n_fx * X * X * Z2_inv);
		J_pixel_xi(0, 5) = n_fx * Y * Z_inv;
	
		J_pixel_xi(1, 0) = 0;
		J_pixel_xi(1, 1) = -n_fy * Z_inv;
		J_pixel_xi(1, 2) = n_fy * Y * Z2_inv;
		J_pixel_xi(1, 3) = n_fy + n_fy * Y * Y * Z2_inv;
		J_pixel_xi(1, 4) = -n_fy * X * Y * Z2_inv;
		J_pixel_xi(1, 5) = -n_fy * X * Z_inv;

		int pixel_counter = 0;

		for (int x = -half_patch_size; x < half_patch_size; x++)
			for (int y = -half_patch_size; y < half_patch_size; y++, ++pixel_counter)
			{
				double value = GetPixelValue(img_ref, u_ref + x, v_ref + y);
				ref_patch_cache_(feature_counter,pixel_counter) = value;

				Vector2d J_img_pixel;

				J_img_pixel = Vector2d(0.5 * (GetPixelValue(img_ref, u_ref+x+1, v_ref+y) - GetPixelValue(img_ref, u_ref+x-1, v_ref+y)),
							0.5 * (GetPixelValue(img_ref, u_ref+x, v_ref+y +1) - GetPixelValue(img_ref, u_ref+x, v_ref+y-1)));
				
				//jacobian_cache_.col(feature_counter*patch_area_ + pixel_counter) = (J_img_pixel.transpose() * J_pixel_xi).transpose();
				jacobian_cache_.col(feature_counter*patch_area_ + pixel_counter) = J_img_pixel.transpose() * J_pixel_xi;
			}
	}
	have_ref_patch_cache_ = true;
}

double Estimator::computeResiduals(
    const cv::Mat &_img1,
    const cv::Mat &_img2,
    const vector<pair<int, Eigen::Matrix<double, 7, 1>>> &_points,
    Sophus::SE3<double> &_T21, const double n_fx, const double n_fy, const double n_cx, const double n_cy, const int level)
{
	cv::Mat img_ref = _img1;
 	cv::Mat img_cur = _img2;	
	double scales[] = {1.0, 0.5, 0.25, 0.125};

	if(have_ref_patch_cache_ == false)
	{
    		precomputeReferencePatches(img_ref, _points, n_fx, n_fy, n_cx, n_cy);
	}
	int chi2 = 0;
	double cost = 0;
	const int border = 3;
	const int half_patch_size = 2;
	const int patch_area_ = 16;
	int feature_counter = 0;
	vector<bool>::iterator visiblity_it = visible_fts_.begin();

	for(auto it = _points.begin(), ite = _points.end(); it!=ite; ++it, ++feature_counter, ++visiblity_it)
       	{
		Feature_in_Frame f_in_fra9((*it).second);
		const Vector3d xyz_ref = f_in_fra9.point;

		if(!*visiblity_it)
      			continue;
		Vector2d puv_cur = world2cam(_T21*xyz_ref);
		const double u_cur = scales[level] * puv_cur(0);
    		const double v_cur = scales[level] * puv_cur(1);

		if(u_cur < 0 || v_cur < 0 || u_cur-border < 0 || v_cur-border < 0 || u_cur+border >= img_cur.cols || v_cur+border >= img_cur.rows)
      			continue;

    		int pixel_counter = 0;

		for (int x = -half_patch_size; x < half_patch_size; x++)
			for (int y = -half_patch_size; y < half_patch_size; y++, ++pixel_counter)
			{
				double value = GetPixelValue(img_cur, u_cur + x, v_cur + y);
				double res = value - ref_patch_cache_(feature_counter,pixel_counter);
				const Vector6d J = jacobian_cache_.col(feature_counter*patch_area_ + pixel_counter);
				H_.noalias() += J*J.transpose();
          			Jres_.noalias() -= J*res;
				cost += res*res;
				chi2++;
			}

	}
	return cost/chi2;
}

void Estimator::DirectPoseEstimation(
    const cv::Mat &_img1,
    const cv::Mat &_img2,
    const vector<pair<int, Eigen::Matrix<double, 7, 1>>> &_points,
    Sophus::SE3<double> &_T21, const double n_fx, const double n_fy, const double n_cx, const double n_cy, const int level)
{
	Sophus::SE3<double> T21_old;
	T21_old = _T21;
	int iterations = 10;
	double cost = 0, lastCost = 0;
	for (int iter = 0; iter < iterations; iter++)
	{
	    	H_.setZero();
    		Jres_.setZero();
		cost = computeResiduals(_img1, _img2, _points, _T21, n_fx, n_fy, n_cx, n_cy, level);

		// solve update and put it into estimation
            	Vector6d update = H_.ldlt().solve(Jres_);

            	if (std::isnan(update[0])) {
                	// sometimes occurred when we have a black or white patch and H is irreversible
                	//cout << "update is nan" << endl;
                	break;
            	}
            	if (iter > 0 && cost > lastCost) {
                	//cout << "cost increased: " << cost << ", " << lastCost << endl;
                	break;
            	}
           	
		_T21 = _T21 * Sophus::SE3<double>::exp(-update) ;
            	lastCost = cost;
		T21_old = _T21;

		if (update.norm() < 1e-3) {
                	//cout << "update is converge" << endl;
                	break;
            	}
	}
}

void Estimator::DirectPoseEstimationMultiLayer(const cv::Mat &img1, const cv::Mat &img2, const vector<pair<int, Eigen::Matrix<double, 7, 1>>> &points, Sophus::SE3<double> &T21) 
{
	// parameters
	int pyramids = 4;
	double pyramid_scale = 0.5;
	double scales[] = {1.0, 0.5, 0.25, 0.125};

	const int patch_halfsize_ = 2;
        const int patch_size_ = 2*patch_halfsize_;//4
        const int patch_area_ = patch_size_*patch_size_;//16

	// create pyramids
	vector<cv::Mat> pyr1, pyr2; // image pyramids
	for (int i = 0; i < pyramids; i++) 
	{
		if (i == 0) 
		{
			pyr1.push_back(img1);
			pyr2.push_back(img2);
		} else 
		{
			cv::Mat img1_pyr, img2_pyr;
			cv::resize(pyr1[i - 1], img1_pyr,
					cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
			cv::resize(pyr2[i - 1], img2_pyr,
					cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
			pyr1.push_back(img1_pyr);
			pyr2.push_back(img2_pyr);
		}
	}
	int num = (int)points.size();
	ref_patch_cache_.resize(num, Eigen::NoChange);
        jacobian_cache_.resize(Eigen::NoChange, num*patch_area_);
        visible_fts_.resize(num, false);

	double m_fx=4.616e+02;
	double m_fy=4.603e+02;
	double m_cx=3.630e+02;
	double m_cy=2.481e+02;

	for (int level = pyramids - 1; level >= 0; level--) 
	{
		vector<pair<int, Eigen::Matrix<double, 7, 1>>> points_pyr; // set the keypoints in this pyramid level
		for (auto &px: points) 
		{
			Feature_in_Frame f_in_fra9(px.second);
			const double u_ref = scales[level] * f_in_fra9.uv(0);
    			const double v_ref = scales[level] * f_in_fra9.uv(1);
			Matrix<double, 7, 1> xyz_uv_velocity;
			xyz_uv_velocity << f_in_fra9.point, u_ref, v_ref, f_in_fra9.velocity;
			points_pyr.push_back(make_pair(px.first,xyz_uv_velocity));  
		}
	
		// scale fx, fy, cx, cy in different pyramid levels
		double n_fx = m_fx * scales[level];
		double n_fy = m_fy * scales[level];
		double n_cx = m_cx * scales[level];
		double n_cy = m_cy * scales[level];

		ref_patch_cache_.setZero();
		jacobian_cache_.setZero();
    		have_ref_patch_cache_ = false;

		DirectPoseEstimation(pyr1[level], pyr2[level], points_pyr, T21, n_fx, n_fy, n_cx, n_cy, level);
	}
	
}

Vector2d Estimator::world2cam(Vector3d xyz)
{
    double dep_j = xyz(2);
    const double u_cur = xyz(0) / dep_j;
    const double v_cur = xyz(1) / dep_j;
    double n_k1=-0.2917;
    double n_k2= 8.228e-02;
    double n_p1=5.333e-05;
    double n_p2=-1.578e-04;
    double n_fx=4.616e+02;
    double n_fy=4.603e+02;
    double n_cx=3.630e+02;
    double n_cy=2.481e+02;
    double mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u,add_distort_u,add_distort_v;
    Vector2d uv;
    mx2_u = u_cur * u_cur;
    my2_u = v_cur * v_cur;
    mxy_u = u_cur * v_cur;
    rho2_u = mx2_u + my2_u;
    rad_dist_u = n_k1 * rho2_u + n_k2 * rho2_u * rho2_u;
    add_distort_u=u_cur+u_cur * rad_dist_u + 2.0 * n_p1 * mxy_u + n_p2 * (rho2_u + 2.0 * mx2_u);
    add_distort_v=v_cur+v_cur * rad_dist_u + 2.0 * n_p2 * mxy_u + n_p1 * (rho2_u + 2.0 * my2_u);
    uv(0)=add_distort_u*n_fx+n_cx;
    uv(1)=add_distort_v*n_fy+n_cy;

    return uv;

}

void Estimator::distortion(const Vector2d& p_u, Vector2d& d_u)
{
    double k1 = -0.2917;
    double k2 = 8.228e-02;;
    double p1 = 5.333e-05;
    double p2 = -1.578e-04;

    double mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;

    mx2_u = p_u(0) * p_u(0);
    my2_u = p_u(1) * p_u(1);
    mxy_u = p_u(0) * p_u(1);
    rho2_u = mx2_u + my2_u;
    rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;
    d_u << p_u(0) * rad_dist_u + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u),
           p_u(1) * rad_dist_u + 2.0 * p2 * mxy_u + p1 * (rho2_u + 2.0 * my2_u);
}

void Estimator::cam2world(const Vector2d& p, Vector3d& P)
{
    double mx_d, my_d, mx_u, my_u;

    double n_fx=4.616e+02;
    double n_fy=4.603e+02;
    double n_cx=3.630e+02;
    double n_cy=2.481e+02;

    double m_inv_K11 = 1.0/n_fx;
    double m_inv_K13 = -n_cx/n_fx;
    double m_inv_K22 = 1.0/n_fy;
    double m_inv_K23 = -n_cy/n_fy;

    // Lift points to normalised plane
    mx_d = m_inv_K11 * p(0) + m_inv_K13;
    my_d = m_inv_K22 * p(1) + m_inv_K23;

    int n = 8;
    Vector2d d_u;
    distortion(Vector2d(mx_d, my_d), d_u);
    // Approximate value
    mx_u = mx_d - d_u(0);
    my_u = my_d - d_u(1);

    for (int i = 1; i < n; ++i)
    {
	distortion(Vector2d(mx_u, my_u), d_u);
	mx_u = mx_d - d_u(0);
	my_u = my_d - d_u(1);
    }

    // Obtain a projective ray
    P << mx_u, my_u, 1.0;
}

bool Estimator::direct_Align2D(const cv::Mat &img1, const cv::Mat &img2,const Vector2d &puv_ref, Vector2d &puv_cur)
{
	Vector2d uv_cur;
	Vector2d last_uv_cur = puv_cur;
	int half_patch_size = 1;
	const int iterations = 10;
	double cost = 0, lastCost = 0;
	for (int iter = 0; iter < iterations; iter++)
	{
		Matrix2d hessian = Matrix2d::Zero();
		Vector2d bias = Vector2d::Zero();
		for(int x=-half_patch_size; x<=half_patch_size; x++)
		{
			for(int y=-half_patch_size; y<=half_patch_size; y++)
			{
				double error = GetPixelValue(img1,puv_ref(0)+x,puv_ref(1)+y) - GetPixelValue(img2,puv_cur(0)+x,puv_cur(1)+y);
				Vector2d J;
				J = -1* Vector2d(0.5 * (GetPixelValue(img2, puv_cur(0)+x+1, puv_cur(1)+y) - GetPixelValue(img2,puv_cur(0)+x-1, puv_cur(1)+y)),
										0.5 * (GetPixelValue(img2, puv_cur(0)+x, puv_cur(1)+y+1) - GetPixelValue(img2, puv_cur(0)+x, puv_cur(1)+y-1)));
				hessian += J * J.transpose();
				bias += -error * J;
				cost += error * error;
			}
		}
		Vector2d update = hessian.ldlt().solve(bias);
		if (std::isnan(update[0])) 
		{
			//cout << "update is nan" << endl;
			break;
		}
		if (iter > 0 && cost > lastCost) 
		{
			//cout << "cost increased: " << cost << ", " << lastCost << endl;
			break;
		}
		puv_cur(0) += update[0];
		puv_cur(1) += update[1];
		lastCost = cost;
		if (update.norm() < 1e-2) 
		{
			//cout << "update is converge" << endl;
			break;
		}
		if (puv_cur(0) < half_patch_size || puv_cur(0) > img2.cols - half_patch_size || puv_cur(1) < half_patch_size || puv_cur(1) > img2.rows - half_patch_size)
		{
			break;
		}			
	}
	uv_cur(0) = puv_cur(0);
	uv_cur(1) = puv_cur(1);
			
	if( fabs(uv_cur(0)-last_uv_cur(0))<=5 && fabs(uv_cur(1)-last_uv_cur(1))<=5 && uv_cur(0)>0 && uv_cur(1)>0 )
	{
		return true;
	}else
	{
		return false;
	}
}

void Estimator::get_image_Pyramid(const cv::Mat &img1, const cv::Mat &img2, vector<cv::Mat> &pyr1,  vector<cv::Mat> &pyr2)
{
	int pyramids = 4;
	double pyramid_scale = 0.5;
	
	for (int i = 0; i < pyramids; i++) 
	{
		if (i == 0) 
		{
			pyr1.push_back(img1);
			pyr2.push_back(img2);
		} else 
		{
			cv::Mat img1_pyr, img2_pyr;
			cv::resize(pyr1[i - 1], img1_pyr,
					cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
			cv::resize(pyr2[i - 1], img2_pyr,
					cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
			pyr1.push_back(img1_pyr);
			pyr2.push_back(img2_pyr);
		}
	}
}

bool Estimator::judge_keyframe(const vector<pair<int, Eigen::Matrix<double, 7, 1>>> &_points)
{   
    double parallax_sum = 0;
    int parallax_num = 0;
    double frame_threshold = 0.05;//0.02174
    int num_point_frame9 = (int)_points.size();
    if(num_point_frame9 <= 50)
    {
	return true;
    }
    for (auto &it_per_id : f_manager.feature)
    {
        if (it_per_id.start_frame <= frame_count - 2 && it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)
        {
                const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    		const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    		double ans = 0;
    		Vector3d p_j = frame_j.point;

    		double u_j = p_j(0);
    		double v_j = p_j(1);

   	 	Vector3d p_i = frame_i.point;
    		Vector3d p_i_comp;

    		//int r_i = frame_count - 2;
    		//int r_j = frame_count - 1;
    		//p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    		p_i_comp = p_i;
    		double dep_i = p_i(2);
    		double u_i = p_i(0) / dep_i;
    		double v_i = p_i(1) / dep_i;
    		double du = u_i - u_j, dv = v_i - v_j;

    		double dep_i_comp = p_i_comp(2);
    		double u_i_comp = p_i_comp(0) / dep_i_comp;
    		double v_i_comp = p_i_comp(1) / dep_i_comp;
    		double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    		ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));
	        parallax_sum += ans;
                parallax_num++;
        }
    }
    if (parallax_num == 0)
    {
        return true;
    }
    else
    {
        return parallax_sum / parallax_num >= frame_threshold;
    }
}

void Estimator::add_into_feature(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &points_frame_10)
{
	double td = 0.0;
	//int a = 0;
	for (auto &id_pts : points_frame_10)
    	{
		FeaturePerFrame f_per_fra(id_pts.second[0].second, td);
		int feature_id = id_pts.first;

		for (auto &it_per_id :f_manager.feature)
		{

			if (it_per_id.feature_id == feature_id)
        		{
            			it_per_id.feature_per_frame.push_back(f_per_fra);
				//a++;
        		}
		}
	}
	//printf("a  : %d \n", a);
}

double Estimator::distance(cv::Point2f &pt1, cv::Point2f &pt2)
{
    //printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

bool Estimator::inBorder(const cv::Mat &image2, cv::Point2f points)
{
   if(points.x < 0 || points.y < 0 || points.x-1 < 0 || points.y-1 < 0 || points.x+1 >= image2.cols || points.y+1 >= image2.rows)
   {
	return false;
   }
    return true;
}
void Estimator::reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void Estimator::reducePoint2f(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> Estimator::goodFeaturesToTrack(const cv::Mat &image1, const cv::Mat &image2, const vector<pair<int, Eigen::Matrix<double, 7, 1>>> &points)
{
    cv::Mat mask;
    mask = cv::Mat(image1.rows, image1.cols, CV_8UC1, cv::Scalar(255));
    vector<pair<int, Eigen::Matrix<double, 7, 1>>> new_points_frame9;
    vector<pair<int, Eigen::Matrix<double, 7, 1>>> new_points_frame10;
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> points_frame_10;
    
    int frame9 = 9;
    double td = 0.0;
    int max_cnt = 150;
    double min_distance = 30;
    int camera_id = 0;
    vector<cv::Point2f> new_points;
    for (auto &it : points)
    {
	Feature_in_Frame f_in_fra9(it.second);
	cv::Point2f uv_cv;
	double u= f_in_fra9.uv(0);
	double v= f_in_fra9.uv(1);
	uv_cv =  cv::Point2f(u,v);

	if (mask.at<uchar>(uv_cv) == 255)
        {
            cv::circle(mask, uv_cv, min_distance, 0, -1);
        }
    }	
    //printf("old points in feature9: %d \n", (int)points.size());

    int n_max_cnt = max_cnt - static_cast<int>(points.size());
    if(n_max_cnt > 0)
    {
    	cv::goodFeaturesToTrack(image1, new_points, n_max_cnt, 0.01, min_distance, mask);

	vector<cv::Point2f> ref_pts_new(new_points.size());
    	vector<cv::Point2f> cur_pts_new;
    	vector<int> ids_new(new_points.size());
    	int r = 0;
    	for (auto &it : new_points)
    	{
		ref_pts_new[r] = cv::Point2f(it.x, it.y);
		int feature_id = featureTracker.n_id++;
		//printf("id of new in feature9: %d \n", feature_id);
 		ids_new[r] = feature_id;
		r++;
    	}
    	vector<uchar> status_new;
    	vector<float> err_new;
    	cv::calcOpticalFlowPyrLK(image1, image2, ref_pts_new, cur_pts_new, status_new, err_new, cv::Size(21, 21), 3); 
    	for (int i = 0; i < int(cur_pts_new.size()); i++)
           	 if (status_new[i] && !inBorder(image2,cur_pts_new[i]))
               	 status_new[i] = 0;
    	reducePoint2f(ref_pts_new, status_new);
   	reducePoint2f(cur_pts_new, status_new);
    	reduceVector(ids_new, status_new);

        for (unsigned int i = 0; i < ref_pts_new.size(); i++)
        {
		Vector3d tmp_p;
		int feature_id = ids_new[i];
		Vector2d ref_pts = Vector2d(ref_pts_new[i].x, ref_pts_new[i].y);
		cam2world(ref_pts, tmp_p);
		double depth = tmp_p(2);
		double x_point = tmp_p(0)/depth;
		double y_point = tmp_p(1)/depth;
		double z_point = tmp_p(2)/depth;
		double velocity_x =  0;
		double velocity_y =  0;
		Matrix<double, 7, 1> xyz_uv_velocity;
		xyz_uv_velocity << x_point, y_point, z_point, ref_pts, velocity_x, velocity_y;
		new_points_frame9.push_back(make_pair(feature_id,xyz_uv_velocity)); 
    	}
    	for (auto &id_pts : new_points_frame9)
    	{
		FeaturePerFrame f_per_fra(id_pts.second, td);
		int feature_id = id_pts.first;
        	f_manager.feature.push_back(FeaturePerId(feature_id, frame9));
        	f_manager.feature.back().feature_per_frame.push_back(f_per_fra);
    	}

	for (unsigned int i = 0; i < cur_pts_new.size(); i++)
    	{
		Vector3d tmp_p;
		int feature_id = ids_new[i];
		Vector2d cur_pts = Vector2d(cur_pts_new[i].x, cur_pts_new[i].y);
		cam2world(cur_pts, tmp_p);
		double depth = tmp_p(2);
		double x_point = tmp_p(0)/depth;
		double y_point = tmp_p(1)/depth;
		double z_point = tmp_p(2)/depth;
		double velocity_x =  0;
		double velocity_y =  0;
		Matrix<double, 7, 1> xyz_uv_velocity;
		xyz_uv_velocity << x_point, y_point, z_point, cur_pts, velocity_x, velocity_y;
		new_points_frame10.push_back(make_pair(feature_id,xyz_uv_velocity));
		points_frame_10[feature_id].emplace_back(camera_id,  xyz_uv_velocity); 
    	}

    }
    //printf("new points in feature9: %d \n", (int)new_points.size());
    
    vector<cv::Point2f> ref_pts_old(points.size());
    vector<cv::Point2f> cur_pts_old;
    vector<int> ids_old(points.size());
    int s = 0;
    for (auto &it : points)
    {
	Feature_in_Frame f_in_fra9(it.second);
	const Vector2d uv = f_in_fra9.uv;
	const int feature_id = it.first;
	ref_pts_old[s] = cv::Point2f(uv(0),uv(1));	
	ids_old[s] = feature_id;
	s++;
    }
    vector<uchar> status_old;
    vector<float> err_old;
    cv::calcOpticalFlowPyrLK(image1, image2, ref_pts_old, cur_pts_old, status_old, err_old, cv::Size(21, 21), 3); 
    for (int i = 0; i < int(cur_pts_old.size()); i++)
            if (status_old[i] && !inBorder(image2,cur_pts_old[i]))
                status_old[i] = 0;
    reducePoint2f(ref_pts_old, status_old);
    reducePoint2f(cur_pts_old, status_old);
    reduceVector(ids_old, status_old);
 
    for (unsigned int i = 0; i < cur_pts_old.size(); i++)
    {
	Vector3d tmp_p;
	int feature_id = ids_old[i];
	Vector2d cur_pts = Vector2d(cur_pts_old[i].x, cur_pts_old[i].y);
	cam2world(cur_pts, tmp_p);
	double depth = tmp_p(2);
	double x_point = tmp_p(0)/depth;
	double y_point = tmp_p(1)/depth;
	double z_point = tmp_p(2)/depth;
	double velocity_x =  0;
	double velocity_y =  0;
	Matrix<double, 7, 1> xyz_uv_velocity;
	xyz_uv_velocity << x_point, y_point, z_point, cur_pts, velocity_x, velocity_y;
	new_points_frame10.push_back(make_pair(feature_id,xyz_uv_velocity)); 
        points_frame_10[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
    }
    
    int num_id = 0;
    for (auto &id_pts : new_points_frame10)
    {
        FeaturePerFrame f_per_fra(id_pts.second, td);
        int feature_id = id_pts.first;

	for (auto &it_per_id :f_manager.feature)
	{
		if (it_per_id.feature_id == feature_id)
        	{
            		it_per_id.feature_per_frame.push_back(f_per_fra);
			num_id++;
        	}
	}

    }
    //printf("number of feature: %d \n", num_id);
    //printf("new points in feature10: %d \n", (int)new_points_frame10.size());
return points_frame_10;

}

map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> Estimator::findMatch_by_Direct(const cv::Mat &img1, const cv::Mat &img2, 
										const vector<pair<int, Eigen::Matrix<double, 7, 1>>> &points, Sophus::SE3<double> &T21, const double header)
{
	vector<cv::Mat> pyr1, pyr2;
	get_image_Pyramid(img1, img2, pyr1, pyr2);
	double dt = header - Headers[9];
	int camera_id = 0;
	//double distance = 0;
	//int distance_num = 0;
	map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> _points_frame_10;
	//vector<Vector2d> ref_pts,cur_pts;
	//vector<int> ids;
	for(auto it = points.begin(), ite = points.end(); it!=ite; ++it)
       	{
		int pyramids = 4;
    		double scales[] = {1.0, 0.5, 0.25, 0.125};		

		Feature_in_Frame f_in_fra10((*it).second);
		const int feature_id = (*it).first;
		const Vector3d xyz_ref = f_in_fra10.point;
		Vector2d puv_ref;
		puv_ref(0) = f_in_fra10.uv(0);
        	puv_ref(1) = f_in_fra10.uv(1);
		Vector2d puv_cur = world2cam(T21*xyz_ref);

		if (puv_cur(0) < 1 || puv_cur(0) > img2.cols - 1 || puv_cur(1) < 1 || puv_cur(1) > img2.rows - 1)
		{		
			continue;
		}	
		bool success_track = true;
		for (int level = pyramids - 1; level >= 0; level--)
		{
			Vector2d uv_ref;
			Vector2d uv_cur;		

			uv_ref(0) = scales[level] * puv_ref(0);
    			uv_ref(1) = scales[level] * puv_ref(1);

			uv_cur(0) = scales[level] * puv_cur(0);
    			uv_cur(1) = scales[level] * puv_cur(1);
			bool res = direct_Align2D(pyr1[level], pyr2[level], uv_ref, uv_cur);
			if(res == true)
			{
				puv_cur(0) = uv_cur(0)/scales[level];
				puv_cur(1) = uv_cur(1)/scales[level];
			}else
			{	
				success_track = false;
				break;
			}
		}
		if(success_track == true)
		{
			//cur_pts.push_back(puv_cur);
			//ref_pts.push_back(puv_ref);
			//ids.push_back(feature_id);

			Vector3d point_j;
			Vector3d point_i;
			//double du;
			//double dv;
			cam2world(puv_cur,point_j);
			cam2world(puv_ref,point_i);
			Matrix<double, 7, 1> xyz_uv_velocity;
			double velocity_x = (point_j(0) - point_i(0)) / dt;
			double velocity_y = (point_j(1) - point_i(1)) / dt;
			//du = point_j(0) - point_i(0);
			//dv = point_j(1) - point_i(1);
			//distance +=sqrt(du * du + dv * dv);
			//distance_num++;
			xyz_uv_velocity << point_j, puv_cur, velocity_x, velocity_y;

			//printf("velocity_x  : %f \n", velocity_x);
			//printf("velocity_y  : %f \n", velocity_y);
			_points_frame_10[feature_id].emplace_back(camera_id,  xyz_uv_velocity);  
		}
		//printf("success tracking feature: %d \n", num_suc_track);
		//bool res = direct_Align2D(image_in_frame[9], image, puv_ref, puv_cur);
		//if(res == true)
		//{
		//	num_suc_track++;
		//}
	}
	//printf("success tracking feature: %f \n", distance/distance_num);
	/*vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_ref_pts(ref_pts.size());
	for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
		un_cur_pts[i] = cv::Point2f(cur_pts[i].x(),cur_pts[i].y());
		un_ref_pts[i] = cv::Point2f(ref_pts[i].x(),ref_pts[i].y());
	}
	vector<cv::Point2f> reverse_pts;
	vector<uchar> status;
	vector<float> err;
	int a = 0;
	cv::calcOpticalFlowPyrLK(img1, img2, un_ref_pts, reverse_pts, status, err, cv::Size(21, 21), 3); 
	for(size_t i = 0; i < status.size(); i++)
        {
		if(status[i] && distance(un_cur_pts[i], reverse_pts[i]) <= 5)
                {
                    a++;
                }
                
        }
	printf("success    : %d \n", a);
	if (cur_pts.size() >= 8)
    	{
		double FOCAL_LENGTH = 460.0;
		int col = img2.cols;
		int row = img2.rows;
		double F_THRESHOLD = 1.0;
		int success_triangulat = 0;

		vector<cv::Point2d> un_cur_pts(cur_pts.size()), un_ref_pts(ref_pts.size());
        	for (unsigned int i = 0; i < cur_pts.size(); i++)
        	{
           	 	Vector3d tmp_p;
	    		cam2world(cur_pts[i], tmp_p);
            		tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            		tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            		un_cur_pts[i] = cv::Point2d(tmp_p.x(), tmp_p.y());

	    		Vector3d tmp_q;
            		cam2world(ref_pts[i], tmp_q);
            		tmp_q.x() = FOCAL_LENGTH * tmp_q.x() / tmp_q.z() + col / 2.0;
            		tmp_q.y() = FOCAL_LENGTH * tmp_q.y() / tmp_q.z() + row / 2.0;
            		un_ref_pts[i] = cv::Point2d(tmp_q.x(), tmp_q.y());
        	}
		vector<uchar> status;
        	cv::findFundamentalMat(un_cur_pts, un_ref_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        	int size = cur_pts.size();
		int j = 0;
		for (int i = 0; i < size; i++)
		{
        		if (status[i])
			{
				success_triangulat++;
				cur_pts[j++] = cur_pts[i];
				ref_pts[j++] = ref_pts[i];
				ids[j++] = ids[i];
			}
			cur_pts.resize(j);
			ref_pts.resize(j);
			ids.resize(j);
		}
		//printf("total    : %d \n", size);
		//printf("success  : %d \n", success_triangulat);
	}
	for (unsigned int i = 0; i < cur_pts.size(); i++)
	{
		const int feature_id = ids[i];		
		Vector3d point_j;
		Vector3d point_i;
		cam2world(cur_pts[i],point_j);
		cam2world(ref_pts[i],point_i);
		Matrix<double, 7, 1> xyz_uv_velocity;
		double velocity_x = (point_j(0) - point_i(0)) / dt;
		double velocity_y = (point_j(1) - point_i(1)) / dt;
		xyz_uv_velocity << point_j, cur_pts[i], velocity_x, velocity_y;

		printf("velocity_x  : %f \n", velocity_x);
		printf("velocity_y  : %f \n", velocity_y);

		_points_frame_10.push_back(make_pair(feature_id,xyz_uv_velocity));  
	}*/
	
	return _points_frame_10;
}

void Estimator::processImage(const cv::Mat &image, const double header)
{
    //ROS_DEBUG("new image coming ------------------------------------------");
    //ROS_DEBUG("Adding feature points %lu", image.size());
    //it_per_id.solve_flag == 1
    vector<pair<int, Eigen::Matrix<double, 7, 1>>> points_frame_9;
    for (auto &it_per_id :f_manager.feature)
    {
	if (it_per_id.start_frame <= WINDOW_SIZE - 2 && it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1  >= WINDOW_SIZE - 1 && it_per_id.estimated_depth > 0)
	{
		const FeaturePerFrame &frame_9 = it_per_id.feature_per_frame[WINDOW_SIZE - 1 - it_per_id.start_frame];
		const int feature_id = it_per_id.feature_id;
		//int camera_id = 0;
				
		Vector3d point = frame_9.point;
		Vector2d p_uv = frame_9.uv;
		Vector2d p_velocity = frame_9.velocity;

		int imu_i = it_per_id.start_frame;
        	Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
        	Vector3d w_pts_i = Rs[imu_i] * (ric[0] * pts_i + tic[0]) + Ps[imu_i];

		Matrix3d R1;
        	Vector3d P1;
        	R1 = Rs[WINDOW_SIZE - 1] * ric[0];
        	P1 = Ps[WINDOW_SIZE - 1] + Rs[WINDOW_SIZE - 1] * tic[0];

		Vector3d pts_j = R1.transpose() * (w_pts_i - P1);
		double dep_j = pts_j(2);
		Vector3d point_j = point * dep_j;

		Matrix<double, 7, 1> xyz_uv_velocity;
		xyz_uv_velocity << point_j, p_uv, p_velocity;

		points_frame_9.push_back(make_pair(feature_id,xyz_uv_velocity));   
	}	
    } 
    //printf("input feature in frame9: %d \n", (int)points_frame_9.size());

    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    if (solver_flag == INITIAL)
    {
    	//map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
	//TicToc time_point;    	
	featureFrame = featureTracker.trackImage(header, image);
    	if (f_manager.addFeatureCheckParallax(frame_count, featureFrame, td))
    	{
       	 	marginalization_flag = MARGIN_OLD;
        	//printf("keyframe\n");
    	}
   	 else
    	{
       	 	marginalization_flag = MARGIN_SECOND_NEW;
        	//printf("non-keyframe\n");
    	}
	//printf("temporal optical flow costs: %fms", time_point.toc());
    }
    else
    {
	TicToc time_direct;
	num_img++;	
	bool keyframe = judge_keyframe(points_frame_9);
	if(keyframe == false)
	{
		//num_direct++;		
		//TicToc time_direct01;	
		marginalization_flag = MARGIN_SECOND_NEW;    		
		Sophus::SE3<double> T21;
		DirectPoseEstimationMultiLayer(image_in_frame[9], image, points_frame_9, T21);
		//map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
		featureFrame = findMatch_by_Direct(image_in_frame[9], image, points_frame_9, T21, header);
		add_into_feature(featureFrame);
		//printf("non keyframe: %d \n", (int)featureFrame.size());
		//time_direct += time_direct01.toc();
		//printf("time_direct:       %fms \n", time_direct);
		//printf(" num_direct:       %d \n", num_direct);
	}
	else
	{
		//num_optic++;		
		//TicToc time_direct02;
		marginalization_flag = MARGIN_OLD;
		featureFrame = goodFeaturesToTrack(image_in_frame[9], image, points_frame_9);
		//time_optic += time_direct02.toc();
		//printf("time_optical:      %fms \n", time_optic);
		//printf(" num_optical:       %d \n", num_optic);
	}
	time_img += time_direct.toc();
	printf("time_process:       %fms \n", time_img);
	printf("nume_process:       %d \n", num_img);
	//printf("temporal direct track costs: %fms", time_direct.toc());
    } 

    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    //ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());

    Headers[frame_count] = header;
    image_in_frame.push_back(image.clone());
    //cout << "The size of image_in_frame is " << image_in_frame.size() << endl;
  
    ImageFrame imageframe(featureFrame, header);
    imageframe.pre_integration = tmp_pre_integration;
    all_image_frame.insert(make_pair(header, imageframe));
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    if(ESTIMATE_EXTRINSIC == 2)
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }

    if (solver_flag == INITIAL)
    {
        // monocular + IMU initilization
        if (!STEREO && USE_IMU)
        {
            if (frame_count == WINDOW_SIZE)
            {
                bool result = false;
                if(ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1)
                {
                    result = initialStructure();
                    initial_timestamp = header;   
                }
                if(result)
                {
                    solver_flag = NON_LINEAR;
                    optimization();
                    slideWindow();
                    ROS_INFO("Initialization finish!");
                }
                else
                    slideWindow();
            }
        }

        // stereo + IMU initilization
        if(STEREO && USE_IMU)
        {
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            if (frame_count == WINDOW_SIZE)
            {
                map<double, ImageFrame>::iterator frame_it;
                int i = 0;
                for (frame_it = all_image_frame.begin(); frame_it != all_image_frame.end(); frame_it++)
                {
                    frame_it->second.R = Rs[i];
                    frame_it->second.T = Ps[i];
                    i++;
                }
		//visualInitialAlign();
                solveGyroscopeBias(all_image_frame, Bgs);
		
                for (int i = 0; i <= WINDOW_SIZE; i++)
                {
                    pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
                }

		solver_flag = NON_LINEAR;
                optimization();
                slideWindow();
                ROS_INFO("Initialization finish!");
            }
        }

        // stereo only initilization
        if(STEREO && !USE_IMU)
        {
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            optimization();

            if(frame_count == WINDOW_SIZE)
            {
                solver_flag = NON_LINEAR;
                slideWindow();
                ROS_INFO("Initialization finish!");
            }
        }

        if(frame_count < WINDOW_SIZE)
        {
            frame_count++;
            int prev_frame = frame_count - 1;
            Ps[frame_count] = Ps[prev_frame];
            Vs[frame_count] = Vs[prev_frame];
            Rs[frame_count] = Rs[prev_frame];
            Bas[frame_count] = Bas[prev_frame];
            Bgs[frame_count] = Bgs[prev_frame];
        }

    }
    else
    {
        TicToc t_solve;
        if(!USE_IMU)
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
        f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
        optimization();
        set<int> removeIndex;
        outliersRejection(removeIndex);
        f_manager.removeOutlier(removeIndex);
        if (! MULTIPLE_THREAD)
        {
            featureTracker.removeOutliers(removeIndex);
            predictPtsInNextFrame();
        }
            
        ROS_DEBUG("solver costs: %fms", t_solve.toc());

        if (failureDetection())
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            return;
        }

        slideWindow();
        f_manager.removeFailures();
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
        updateLatestStates();
    }  
}

bool Estimator::initialStructure()
{
    TicToc t_sfm;
    //check imu observibility
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        //ROS_WARN("IMU variation %f!", var);
        if(var < 0.25)
        {
            ROS_INFO("IMU excitation not enouth!");
            //return false;
        }
    }
    // global sfm
    Quaterniond Q[frame_count + 1];
    Vector3d T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;
    for (auto &it_per_id : f_manager.feature)
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    } 
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }
    GlobalSFM sfm;
    if(!sfm.construct(frame_count + 1, Q, T, l,
              relative_R, relative_T,
              sfm_f, sfm_tracked_points))
    {
        ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    //solve pnp for all frame
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin( );
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if((frame_it->first) == Headers[i])
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if((frame_it->first) > Headers[i])
        {
            i++;
        }
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = - R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if(it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);     
        if(pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }
    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }

}


bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;
    //solve scale
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if(!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    // change state
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i]].R;
        Vector3d Pi = all_image_frame[Headers[i]].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i]].is_key_frame = true;
    }

    double s = (x.tail<1>())(0);
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }

    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose()); 

    f_manager.clearDepth();
    f_manager.triangulate(frame_count, Ps, Rs, tic, ric);

    return true;
}
bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        if(USE_IMU)
        {
            para_SpeedBias[i][0] = Vs[i].x();
            para_SpeedBias[i][1] = Vs[i].y();
            para_SpeedBias[i][2] = Vs[i].z();

            para_SpeedBias[i][3] = Bas[i].x();
            para_SpeedBias[i][4] = Bas[i].y();
            para_SpeedBias[i][5] = Bas[i].z();

            para_SpeedBias[i][6] = Bgs[i].x();
            para_SpeedBias[i][7] = Bgs[i].y();
            para_SpeedBias[i][8] = Bgs[i].z();
        }
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }


    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);

    para_Td[0][0] = td;
}

void Estimator::double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }

    if(USE_IMU)
    {
        Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                          para_Pose[0][3],
                                                          para_Pose[0][4],
                                                          para_Pose[0][5]).toRotationMatrix());
        double y_diff = origin_R0.x() - origin_R00.x();
        //TODO
        Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
        if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
        {
            ROS_DEBUG("euler singular point!");
            rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                           para_Pose[0][3],
                                           para_Pose[0][4],
                                           para_Pose[0][5]).toRotationMatrix().transpose();
        }

        for (int i = 0; i <= WINDOW_SIZE; i++)
        {

            Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
            
            Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                    para_Pose[i][1] - para_Pose[0][1],
                                    para_Pose[i][2] - para_Pose[0][2]) + origin_P0;


                Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                            para_SpeedBias[i][1],
                                            para_SpeedBias[i][2]);

                Bas[i] = Vector3d(para_SpeedBias[i][3],
                                  para_SpeedBias[i][4],
                                  para_SpeedBias[i][5]);

                Bgs[i] = Vector3d(para_SpeedBias[i][6],
                                  para_SpeedBias[i][7],
                                  para_SpeedBias[i][8]);
            
        }
    }
    else
    {
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            Rs[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
            
            Ps[i] = Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
        }
    }

    if(USE_IMU)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            tic[i] = Vector3d(para_Ex_Pose[i][0],
                              para_Ex_Pose[i][1],
                              para_Ex_Pose[i][2]);
            ric[i] = Quaterniond(para_Ex_Pose[i][6],
                                 para_Ex_Pose[i][3],
                                 para_Ex_Pose[i][4],
                                 para_Ex_Pose[i][5]).toRotationMatrix();
        }
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);

    if(USE_IMU)
        td = para_Td[0][0];

}

bool Estimator::failureDetection()
{
    return false;
    if (f_manager.last_track_num < 2)
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        //ROS_INFO(" big translation");
        //return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        //ROS_INFO(" big z translation");
        //return true; 
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}

void Estimator::optimization()
{
    TicToc t_whole, t_prepare;
    vector2double();

    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    //loss_function = NULL;
    loss_function = new ceres::HuberLoss(1.0);
    //loss_function = new ceres::CauchyLoss(1.0 / FOCAL_LENGTH);
    //ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
    for (int i = 0; i < frame_count + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        if(USE_IMU)
            problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    if(!USE_IMU)
        problem.SetParameterBlockConstant(para_Pose[0]);

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        if ((ESTIMATE_EXTRINSIC && frame_count == WINDOW_SIZE && Vs[0].norm() > 0.2) || openExEstimation)
        {
            //ROS_INFO("estimate extinsic param");
            openExEstimation = 1;
        }
        else
        {
            //ROS_INFO("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
    }
    problem.AddParameterBlock(para_Td[0], 1);

    if (!ESTIMATE_TD || Vs[0].norm() < 0.2)
        problem.SetParameterBlockConstant(para_Td[0]);

    if (last_marginalization_info && last_marginalization_info->valid)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }
    if(USE_IMU)
    {
        for (int i = 0; i < frame_count; i++)
        {
            int j = i + 1;
            if (pre_integrations[j]->sum_dt > 10.0)
                continue;
            IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
            problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
        }
    }

    int f_m_cnt = 0;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;
 
        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;
                ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
            }

            if(STEREO && it_per_frame.is_stereo)
            {                
                Vector3d pts_j_right = it_per_frame.pointRight;
                if(imu_i != imu_j)
                {
                    ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                }
                else
                {
                    ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem.AddResidualBlock(f, loss_function, para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                }
               
            }
            f_m_cnt++;
        }
    }

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    //printf("prepare for ceres: %f \n", t_prepare.toc());

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    //printf("solver costs: %f \n", t_solver.toc());

    double2vector();
    //printf("frame_count: %d \n", frame_count);

    if(frame_count < WINDOW_SIZE)
        return;

    TicToc t_whole_marginalization;
    if (marginalization_flag == MARGIN_OLD)
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();

        if (last_marginalization_info && last_marginalization_info->valid)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        if(USE_IMU)
        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                           vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                           vector<int>{0, 1});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (it_per_id.used_num < 4)
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if(imu_i != imu_j)
                    {
                        Vector3d pts_j = it_per_frame.point;
                        ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                        vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                        vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    if(STEREO && it_per_frame.is_stereo)
                    {
                        Vector3d pts_j_right = it_per_frame.pointRight;
                        if(imu_i != imu_j)
                        {
                            ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                           vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                                                                           vector<int>{0, 4});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                        else
                        {
                            ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                           vector<double *>{para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                                                                           vector<int>{2});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                    }
                }
            }
        }

        TicToc t_pre_margin;
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());
        
        TicToc t_margin;
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            if(USE_IMU)
                addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

        addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
        
    }
    else
    {
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info && last_marginalization_info->valid)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());
            
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    if(USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    if(USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

            
            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
            
        }
    }
    //printf("whole marginalization costs: %f \n", t_whole_marginalization.toc());
    //printf("whole time for ceres: %f \n", t_whole.toc());
}

void Estimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD)
    {
        double t_0 = Headers[0];
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Headers[i] = Headers[i + 1];
		//image_in_frame.erase(image_in_frame.begin());
		image_in_frame[i] = image_in_frame[i+1];

                Rs[i].swap(Rs[i + 1]);
                Ps[i].swap(Ps[i + 1]);
                if(USE_IMU)
                {
                    std::swap(pre_integrations[i], pre_integrations[i + 1]);

                    dt_buf[i].swap(dt_buf[i + 1]);
                    linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                    angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                    Vs[i].swap(Vs[i + 1]);
                    Bas[i].swap(Bas[i + 1]);
                    Bgs[i].swap(Bgs[i + 1]);
                }
            }
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
	    //image_in_frame[WINDOW_SIZE] = image_in_frame[WINDOW_SIZE - 1];
	    image_in_frame.pop_back();

            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];

            if(USE_IMU)
            {
                Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
                Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
                Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }

            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                all_image_frame.erase(all_image_frame.begin(), it_0);
            }
            slideWindowOld();
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            Headers[frame_count - 1] = Headers[frame_count];
	    image_in_frame[frame_count - 1] = image_in_frame[frame_count];

	    //swap(image_in_frame[frame_count - 1],image_in_frame[frame_count]); 
 	    image_in_frame.pop_back();
	    //image_in_frame[WINDOW_SIZE].clear();
            Ps[frame_count - 1] = Ps[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];

            if(USE_IMU)
            {
                for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
                {
                    double tmp_dt = dt_buf[frame_count][i];
                    Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                    Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                    pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                    dt_buf[frame_count - 1].push_back(tmp_dt);
                    linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                    angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
                }

                Vs[frame_count - 1] = Vs[frame_count];
                Bas[frame_count - 1] = Bas[frame_count];
                Bgs[frame_count - 1] = Bgs[frame_count];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }
            slideWindowNew();
        }
    }
}

void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}

void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager.removeBack();
}


void Estimator::getPoseInWorldFrame(Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[frame_count];
    T.block<3, 1>(0, 3) = Ps[frame_count];
}

void Estimator::getPoseInWorldFrame(int index, Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[index];
    T.block<3, 1>(0, 3) = Ps[index];
}

void Estimator::predictPtsInNextFrame()
{
    //printf("predict pts in next frame\n");
    if(frame_count < 2)
        return;
    // predict next pose. Assume constant velocity motion
    Eigen::Matrix4d curT, prevT, nextT;
    getPoseInWorldFrame(curT);
    getPoseInWorldFrame(frame_count - 1, prevT);
    nextT = curT * (prevT.inverse() * curT);
    map<int, Eigen::Vector3d> predictPts;

    for (auto &it_per_id : f_manager.feature)
    {
        if(it_per_id.estimated_depth > 0)
        {
            int firstIndex = it_per_id.start_frame;
            int lastIndex = it_per_id.start_frame + it_per_id.feature_per_frame.size() - 1;
            //printf("cur frame index  %d last frame index %d\n", frame_count, lastIndex);
            if((int)it_per_id.feature_per_frame.size() >= 2 && lastIndex == frame_count)
            {
                double depth = it_per_id.estimated_depth;
                Vector3d pts_j = ric[0] * (depth * it_per_id.feature_per_frame[0].point) + tic[0];
                Vector3d pts_w = Rs[firstIndex] * pts_j + Ps[firstIndex];
                Vector3d pts_local = nextT.block<3, 3>(0, 0).transpose() * (pts_w - nextT.block<3, 1>(0, 3));
                Vector3d pts_cam = ric[0].transpose() * (pts_local - tic[0]);
                int ptsIndex = it_per_id.feature_id;
                predictPts[ptsIndex] = pts_cam;
            }
        }
    }
    featureTracker.setPrediction(predictPts);
    //printf("estimator output %d predict pts\n",(int)predictPts.size());
}

double Estimator::reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                 Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj, 
                                 double depth, Vector3d &uvi, Vector3d &uvj)
{
    Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
    Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
    Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
    double rx = residual.x();
    double ry = residual.y();
    return sqrt(rx * rx + ry * ry);
}

void Estimator::outliersRejection(set<int> &removeIndex)
{
    //return;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        double err = 0;
        int errCnt = 0;
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;
        feature_index ++;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;
        double depth = it_per_id.estimated_depth;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;             
                double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], 
                                                    Rs[imu_j], Ps[imu_j], ric[0], tic[0],
                                                    depth, pts_i, pts_j);
                err += tmp_error;
                errCnt++;
                //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
            }
            // need to rewrite projecton factor.........
            if(STEREO && it_per_frame.is_stereo)
            {
                
                Vector3d pts_j_right = it_per_frame.pointRight;
                if(imu_i != imu_j)
                {            
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], 
                                                        Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                        depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }
                else
                {
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], 
                                                        Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                        depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }       
            }
        }
        double ave_err = err / errCnt;
        if(ave_err * FOCAL_LENGTH > 3)
            removeIndex.insert(it_per_id.feature_id);

    }
}

void Estimator::fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity)
{
    double dt = t - latest_time;
    latest_time = t;
    Eigen::Vector3d un_acc_0 = latest_Q * (latest_acc_0 - latest_Ba) - g;
    Eigen::Vector3d un_gyr = 0.5 * (latest_gyr_0 + angular_velocity) - latest_Bg;
    latest_Q = latest_Q * Utility::deltaQ(un_gyr * dt);
    Eigen::Vector3d un_acc_1 = latest_Q * (linear_acceleration - latest_Ba) - g;
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    latest_P = latest_P + dt * latest_V + 0.5 * dt * dt * un_acc;
    latest_V = latest_V + dt * un_acc;
    latest_acc_0 = linear_acceleration;
    latest_gyr_0 = angular_velocity;
}

void Estimator::updateLatestStates()
{
    latest_time = Headers[frame_count] + td;
    latest_P = Ps[frame_count];
    latest_Q = Rs[frame_count];
    latest_V = Vs[frame_count];
    latest_Ba = Bas[frame_count];
    latest_Bg = Bgs[frame_count];
    latest_acc_0 = acc_0;
    latest_gyr_0 = gyr_0;
    mBuf.lock();
    queue<pair<double, Eigen::Vector3d>> tmp_accBuf = accBuf;
    queue<pair<double, Eigen::Vector3d>> tmp_gyrBuf = gyrBuf;
    while(!tmp_accBuf.empty())
    {
        double t = tmp_accBuf.front().first;
        Eigen::Vector3d acc = tmp_accBuf.front().second;
        Eigen::Vector3d gyr = tmp_gyrBuf.front().second;
        fastPredictIMU(t, acc, gyr);
        tmp_accBuf.pop();
        tmp_gyrBuf.pop();
    }
    mBuf.unlock();
}
