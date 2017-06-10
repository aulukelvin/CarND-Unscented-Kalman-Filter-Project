#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  n_x_ = 5;
  n_aug_ = 7;
  
  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  
  //create augmented mean vector
  x_aug = VectorXd(n_aug_);
  
  //create augmented state covariance
  P_aug = MatrixXd(n_aug_, n_aug_);
  
  //create sigma point matrix
  Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  
  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.35;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.035;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.35;

  is_initialized_ = false;

  lambda_ = 3 - n_aug_;

  P_ <<   1,0,0,0,0,
          0,1,0,0,0,
          0,0,1,0,0,
          0,0,0,1,0,
          0,0,0,0,1;
  
  
  weights_ = VectorXd(2*n_aug_+1);
  weights_(0) = lambda_/(lambda_+n_aug_);
  for (int i=1; i<2*n_aug_+1; i++) {
    weights_(i) = 0.5/(n_aug_+lambda_);
  }
  
  n_z_radar_ = 3;
  n_z_lidar = 2;
  
  //add measurement noise covariance matrix
  R_radar = MatrixXd(n_z_radar_,n_z_radar_);
  R_radar <<    std_radr_*std_radr_, 0, 0,
                  0, std_radphi_*std_radphi_, 0,
                  0, 0,std_radrd_*std_radrd_;
  
  R_lidar = MatrixXd(n_z_lidar,n_z_lidar);
  R_lidar <<    std_laspx_*std_laspx_, 0,
                  0, std_laspy_*std_laspy_;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  //if not initialized then initialize
  if (!is_initialized_) {
    /**
     * Initialize the state ekf_.x_ with the first measurement.
     */
    if (meas_package.sensor_type_ == meas_package.LASER) {
      double py = meas_package.raw_measurements_(1);
      double px = meas_package.raw_measurements_(0);
      double yaw = atan2(py, px);
      
      yaw = Tools::Normalize(yaw);
      
      x_ << meas_package.raw_measurements_, 0.0, yaw, 0.0;
      cout << "x_  " << x_  << endl;
    } else {
      double rho = meas_package.raw_measurements_(0);
      double yaw = meas_package.raw_measurements_(1);
      double rho_dot = meas_package.raw_measurements_(2);
      double px = rho * cos(yaw);
      double py = rho * sin(yaw);
      x_  << px, py, rho_dot, yaw, 0;
      cout << "x_  " << x_  << endl;
    }
    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    cout << "UKF initialized." << endl;
    return;
  }

  //else pridict
  Prediction(meas_package.timestamp_ - time_us_);
  time_us_ = meas_package.timestamp_;
  
  
  //then update
  if (meas_package.sensor_type_ == meas_package.LASER) {
    if (use_laser_)
      UpdateLidar(meas_package);
  }else {
    if (use_radar_)
      UpdateRadar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  //create sigma point matrix
  AugmentedSigmaPoints();
  SigmaPointPrediction(delta_t/1000000);
  PredictMeanAndCovariance();
}

void UKF::AugmentedSigmaPoints() {
  //create augmented mean state
  x_aug.head(n_x_) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;
  
  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_,n_x_) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;
  
  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();
  
  //create augmented sigma points
  Xsig_aug.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; i++)
  {
    Xsig_aug.col(i+1)       = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }
  
  //print result
//  std::cout << "Xsig_aug = " << std::endl << Xsig_aug << std::endl;
}

void UKF::SigmaPointPrediction(double delta_t) {
  //predict sigma points
  for (int i = 0; i< 2*n_aug_+1; i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);
    
    //predicted state values
    double px_p, py_p;
    
    //avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
      py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else {
      px_p = p_x + v*delta_t*cos(yaw);
      py_p = p_y + v*delta_t*sin(yaw);
    }
    
    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;
    
    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;
    
    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;
    
    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }

  //print result
//  std::cout << "Xsig_pred = " << std::endl << Xsig_pred_ << std::endl;
}

void UKF::PredictMeanAndCovariance() {
  //create vector for predicted state
  VectorXd x = VectorXd(n_x_);
  
  //create covariance matrix for prediction
  MatrixXd P = MatrixXd(n_x_, n_x_);
  
  //predicted state mean
  x.fill(0.0);
  
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x = x+ weights_(i) * Xsig_pred_.col(i);
  }
  
  //predicted state covariance matrix
  P.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x;
    //angle normalization
    x_diff(3) = Tools::Normalize(x_diff(3));
    
    P = P + weights_(i) * x_diff * x_diff.transpose() ;
  }

  //print result
//  std::cout << "Predicted state" << std::endl;
//  std::cout << x << std::endl;
//  std::cout << "Predicted covariance matrix" << std::endl;
//  std::cout << P << std::endl;
  
  //write result
  x_ = x;
  P_ = P;
}
/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  //transform sigma points into measurement space ///// begin
  Zsig_ = MatrixXd(n_z_lidar, 2 * n_aug_ +1);
  for (int i = 0; i < 2 * n_aug_ + 1; i++){  //2n+1 simga points
    // measurement model
    Zsig_(0, i) = Xsig_pred_(0, i);  //px
    Zsig_(1, i) = Xsig_pred_(1, i);  //py
  }
  
  //mean predicted measurement
  z_pred_ = VectorXd(n_z_lidar);
  z_pred_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred_ = z_pred_ + weights_(i) * Zsig_.col(i);
  }
  
  //measurement covariance matrix S
  S_ = MatrixXd(n_z_lidar, n_z_lidar);
  S_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig_.col(i) - z_pred_;
    S_ = S_ + weights_(i) * z_diff * z_diff.transpose();
  }
  
  //add measurement noise covariance matrix
  S_ = S_ + R_lidar;
  //transform sigma points into measurement space ///// end
  
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z_lidar);
  Tc.fill(0.0);
  
  //calculate cross correlation matrix
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    
    //residual
    VectorXd z_diff = Zsig_.col(i) - z_pred_;
    
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    x_diff(3) = Tools::Normalize(x_diff(3));
    
    //Cross-correlation matrix
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }
  
  //Kalman gain K;
  MatrixXd K = Tc * S_.inverse();
  
  //actual measurement
  VectorXd z = VectorXd(n_z_lidar);
  z << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1];// 0.0, 0.0;
  
  //residual
  VectorXd z_diff = z - z_pred_;
  
  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S_ * K.transpose();
  
  //Calculate the lidar NIS.
  double nis = z_diff.transpose() * S_.inverse() * z_diff;
  std::cout << "NIS error= " << nis << std::endl;
  
//  std::cout << "x = \n" << x_ << std::endl;
//  std::cout << "P = \n" << P_ << std::endl;
  
}
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  UKF::PredictRadarMeasurement();
  UKF::UpdateStateRadar(meas_package);
}
/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::PredictRadarMeasurement()  {
  //create matrix for sigma points in measurement space
   Zsig_ = MatrixXd(n_z_radar_, 2 * n_aug_ + 1);
  
  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    
    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);
    
    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;
    
    // measurement model
    Zsig_(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Zsig_(1,i) = atan2(p_y,p_x);                                 //phi
    Zsig_(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
  }
  
  //mean predicted measurement
  z_pred_ = VectorXd(n_z_radar_);
  z_pred_.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
    z_pred_ = z_pred_ + weights_(i) * Zsig_.col(i);
  }
  
  //measurement covariance matrix S
  S_ = MatrixXd(n_z_radar_,n_z_radar_);
  S_.fill(0.0);

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig_.col(i) - z_pred_;
    
    //angle normalization
    z_diff(1) = Tools::Normalize(z_diff(1));
    
    S_ = S_ + weights_(i) * z_diff * z_diff.transpose();
  }
  

    S_ = S_ + R_radar;
  //print result
//  std::cout << "z_pred: " << std::endl << z_pred_ << std::endl;
//  std::cout << "S: " << std::endl << S_ << std::endl;
}

void UKF::UpdateStateRadar(MeasurementPackage meas_package) {
  
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z_radar_);
  
  VectorXd z_ = VectorXd(n_z_radar_);
  
  double rho = meas_package.raw_measurements_(0);
  double yaw = meas_package.raw_measurements_(1);
  double rho_dot = meas_package.raw_measurements_(2);
  z_ << rho, yaw, rho_dot;
  
  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    
    //residual
    VectorXd z_diff = Zsig_.col(i) - z_pred_;
    //angle normalization
    z_diff(1) = Tools::Normalize(z_diff(1));
    
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    z_diff(1) = Tools::Normalize(z_diff(1));
    
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }
  
  //Kalman gain K;
  MatrixXd K = Tc * S_.inverse();
  
  //residual
  VectorXd z_diff = z_ - z_pred_;
  
  //angle normalization
  z_diff(1) = Tools::Normalize(z_diff(1));
  
  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S_*K.transpose();
  
  double nis = z_diff.transpose() * S_.inverse() * z_diff;
  
  //print result
  std::cout << "NIS " << nis << std::endl;
//  std::cout << "Updated state x: " << std::endl << x_ << std::endl;
//  std::cout << "Updated state covariance P: " << std::endl << P_ << std::endl;
  
}
