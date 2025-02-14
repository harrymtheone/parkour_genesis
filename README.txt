#ifndef GO1_GAZEBO_RUNNER_HPP
#define GO1_GAZEBO_RUNNER_HPP

#include <cassert>
#include <csignal>

#include <iostream>
#include <ostream>
#include <ros/ros.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include <std_srvs/Empty.h>
#include <sensor_msgs/JointState.h>
#include <gazebo_msgs/ModelStates.h>
#include <gazebo_msgs/SetModelState.h>
#include <geometry_msgs/Twist.h>

#include "robot_msgs/MotorCommand.h"

#include "loop.hpp"
#include "base_runner.hpp"


template <size_t num_dof>
class Go1GazeboRunner : public BaseRunner<num_dof>
{
    public:
        Go1GazeboRunner(std::string robot_name, ros::NodeHandle &nh);
        ~Go1GazeboRunner(){};

    protected:
        void RobotControl();

        void GetState();
        void SetCommand();

        // loop
        std::shared_ptr<LoopFunc> loop_rl;

        // ros interface
        std::string ros_namespace;
        bool simulation_running = false;
        geometry_msgs::Twist vel;
        geometry_msgs::Pose pose;
        ros::Subscriber model_state_subscriber;
        ros::Subscriber joint_state_subscriber;
        ros::ServiceClient gazebo_set_model_state_client;
        ros::ServiceClient gazebo_pause_physics_client;
        ros::ServiceClient gazebo_unpause_physics_client;
        std::map<std::string, ros::Publisher> joint_publishers;
        std::array<robot_msgs::MotorCommand, num_dof> joint_publishers_commands;
        tf2_ros::TransformBroadcaster tf_broadcaster;
        void ModelStatesCallback(const gazebo_msgs::ModelStates::ConstPtr &msg);
        void JointStatesCallback(const sensor_msgs::JointState::ConstPtr &msg);

        // others
        std::string gazebo_model_name;
        int motiontime = 0;
        std::map<std::string, size_t> sorted_to_original_index;
        std::array<float, num_dof> mapped_joint_positions;
        std::array<float, num_dof> mapped_joint_velocities;
        std::array<float, num_dof> mapped_joint_efforts;
        void MapData(const std::vector<double> &source_data, std::array<float, num_dof> &target_data);
};


template <size_t num_dof>
Go1GazeboRunner<num_dof>::Go1GazeboRunner(std::string robot_name, ros::NodeHandle &nh): BaseRunner<num_dof>(robot_name, nh)
{
    // Due to the fact that the robot_state_publisher sorts the joint names alphabetically,
    // the mapping table is established according to the order defined in the YAML file
    auto sorted_joint_controller_names = this->params.joint_controller_names;
    std::sort(sorted_joint_controller_names.begin(), sorted_joint_controller_names.end());
    for (size_t i = 0; i < this->params.joint_controller_names.size(); ++i)
    {
        this->sorted_to_original_index[sorted_joint_controller_names[i]] = i;
    }

    // publisher
    nh.param<std::string>("ros_namespace", this->ros_namespace, "");
    for (int i = 0; i < num_dof; ++i)
    {
        // joint need to rename as xxx_joint
        this->joint_publishers[this->params.joint_controller_names[i]] =
            nh.advertise<robot_msgs::MotorCommand>(this->ros_namespace + this->params.joint_controller_names[i] + "/command", 10);
    }

    // subscriber
    this->model_state_subscriber = nh.subscribe<gazebo_msgs::ModelStates>("/gazebo/model_states", 10, &Go1GazeboRunner<num_dof>::ModelStatesCallback, this);
    this->joint_state_subscriber = nh.subscribe<sensor_msgs::JointState>(this->ros_namespace + "joint_states", 10, &Go1GazeboRunner<num_dof>::JointStatesCallback, this);

    // service
    nh.param<std::string>("gazebo_model_name", this->gazebo_model_name, "");
    this->gazebo_set_model_state_client = nh.serviceClient<gazebo_msgs::SetModelState>("/gazebo/set_model_state");
    this->gazebo_pause_physics_client = nh.serviceClient<std_srvs::Empty>("/gazebo/pause_physics");
    this->gazebo_unpause_physics_client = nh.serviceClient<std_srvs::Empty>("/gazebo/unpause_physics");

    // loop
    this->loop_control = std::make_shared<LoopFunc>(
        "loop_control", this->params.dt, std::bind(&Go1GazeboRunner<num_dof>::RobotControl, this));
}


template <size_t num_dof>
void Go1GazeboRunner<num_dof>::ModelStatesCallback(const gazebo_msgs::ModelStates::ConstPtr &msg)
{
    for (size_t i = 0; i < msg->name.size(); ++i) {
        if (msg->name[i] == "go1_gazebo") {
            this->vel = msg->twist[i];
            this->pose = msg->pose[i];
        }
    }
}


template <size_t num_dof>
void Go1GazeboRunner<num_dof>::JointStatesCallback(const sensor_msgs::JointState::ConstPtr &msg)
{
    MapData(msg->position, this->mapped_joint_positions);
    MapData(msg->velocity, this->mapped_joint_velocities);
    MapData(msg->effort, this->mapped_joint_efforts);
}


template <size_t num_dof>
void Go1GazeboRunner<num_dof>::MapData(const std::vector<double> &source_data, std::array<float, num_dof> &target_data)
{
    assert(source_data.size() == num_dof);

    for (size_t i = 0; i < num_dof; ++i)
    {
        target_data[i] = source_data[this->sorted_to_original_index[this->params.joint_controller_names[i]]];
    }
}


template <size_t num_dof>
void Go1GazeboRunner<num_dof>::RobotControl()
{
    // reset simulation
    if (this->state_controller.get_control_state() == STATE_RESET_SIMULATION)
    {
        gazebo_msgs::SetModelState set_model_state;
        set_model_state.request.model_state.model_name = this->gazebo_model_name;
        set_model_state.request.model_state.pose.position.z = 1.0;
        set_model_state.request.model_state.reference_frame = "world";
        this->gazebo_set_model_state_client.call(set_model_state);
        this->state_controller.set_control_state(STATE_WAITING);
    }

    // start or stop simulation
    else if (this->state_controller.get_control_state() == STATE_TOGGLE_SIMULATION)
    {
        std_srvs::Empty empty;
        if (this->simulation_running)
        {
            this->gazebo_pause_physics_client.call(empty);
            std::cout << std::endl << LOGGER::INFO << "Simulation Stop" << std::endl;
        }
        else
        {
            this->gazebo_unpause_physics_client.call(empty);
            std::cout << std::endl << LOGGER::INFO << "Simulation Start" << std::endl;
        }
        this->simulation_running = !this->simulation_running;
        this->state_controller.set_control_state(STATE_WAITING);
    }

    // main loop
    if (this->simulation_running)
    {
        this->motiontime++;
        this->GetState();
        this->state_controller.step(this->robot_command, this->robot_state);
        this->SetCommand();
    }
}


template <size_t num_dof>
void Go1GazeboRunner<num_dof>::GetState()
{
    double roll, pitch, yaw;

    {
        std::lock_guard<std::mutex> lock(this->robot_state_mutex);

        if (this->params.framework == "isaacgym")
        {
            this->robot_state.imu.quaternion[3] = this->pose.orientation.w;
            this->robot_state.imu.quaternion[0] = this->pose.orientation.x;
            this->robot_state.imu.quaternion[1] = this->pose.orientation.y;
            this->robot_state.imu.quaternion[2] = this->pose.orientation.z;
        }
        else if (this->params.framework == "isaacsim")
        {
            this->robot_state.imu.quaternion[0] = this->pose.orientation.w;
            this->robot_state.imu.quaternion[1] = this->pose.orientation.x;
            this->robot_state.imu.quaternion[2] = this->pose.orientation.y;
            this->robot_state.imu.quaternion[3] = this->pose.orientation.z;
        }

        this->robot_state.lin_vel[0] = this->vel.linear.x;
        this->robot_state.lin_vel[1] = this->vel.linear.y;
        this->robot_state.lin_vel[2] = this->vel.linear.z;

        this->robot_state.imu.gyroscope[0] = this->vel.angular.x;
        this->robot_state.imu.gyroscope[1] = this->vel.angular.y;
        this->robot_state.imu.gyroscope[2] = this->vel.angular.z;

        // state->imu.accelerometer
        this->robot_state.motor_state.q = this->mapped_joint_positions;
        this->robot_state.motor_state.dq = this->mapped_joint_velocities;
        this->robot_state.motor_state.tauEst = this->mapped_joint_efforts;

        // Set the rotation as a quaternion
        tf2::Matrix3x3({
            this->pose.orientation.x,
            this->pose.orientation.y,
            this->pose.orientation.z,
            this->pose.orientation.w,
        }).getRPY(roll, pitch, yaw);
    }

    geometry_msgs::TransformStamped transformStamped;
    transformStamped.header.stamp = ros::Time::now();
    transformStamped.header.frame_id = "map";
    transformStamped.child_frame_id = "base";

    // Set the translation (x, y, z)
    transformStamped.transform.translation.x = 0;
    transformStamped.transform.translation.y = 0;
    transformStamped.transform.translation.z = 0;

    tf2::Quaternion quat;
    quat.setRPY(roll, pitch, 0);  // Roll, Pitch, Yaw
    transformStamped.transform.rotation.x = quat.x();
    transformStamped.transform.rotation.y = quat.y();
    transformStamped.transform.rotation.z = quat.z();
    transformStamped.transform.rotation.w = quat.w();
    tf_broadcaster.sendTransform(transformStamped);
}


template <size_t num_dof>
void Go1GazeboRunner<num_dof>::SetCommand()
{
    if (this->state_controller.get_running_state() == STATE_WAITING) {
        return;
    }
    
    for (int i = 0; i < num_dof; ++i)
    {
        this->joint_publishers_commands[i].q = this->robot_command.motor_command.q[i];
        this->joint_publishers_commands[i].dq = this->robot_command.motor_command.dq[i];
        this->joint_publishers_commands[i].kp = this->robot_command.motor_command.kp[i];
        this->joint_publishers_commands[i].kd = this->robot_command.motor_command.kd[i];
        this->joint_publishers_commands[i].tau = this->robot_command.motor_command.tau[i];
    }

    for (int i = 0; i < num_dof; ++i)
    {
        this->joint_publishers[this->params.joint_controller_names[i]].publish(this->joint_publishers_commands[i]);
    }
}

#endif
