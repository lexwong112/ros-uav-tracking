#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/Twist.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <std_msgs/String.h>

mavros_msgs::State current_state;
geometry_msgs::PoseStamped pose;
geometry_msgs::PoseStamped current_pose;
geometry_msgs::TwistStamped setVelocity;

int image_width;



enum Flight_Mode {position_mode, velocity_mode};
Flight_Mode flight_mode;

void state_cb(const mavros_msgs::State::ConstPtr& msg)
{
    current_state = *msg;
}

void setPoint(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    geometry_msgs::PoseStamped set_point = *msg;
    pose.pose = set_point.pose;
}

void getCurrentPose(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    current_pose = *msg;
}

void mannualControl(const geometry_msgs::Twist msg)
{
    geometry_msgs::Twist cmd = msg;
    //pose.pose.position.x+=cmd.linear.x/50;
    //pose.pose.position.y+=cmd.angular.z/50;

    setVelocity.twist.linear.x = cmd.linear.x;
    setVelocity.twist.linear.y = cmd.angular.z;
    flight_mode = velocity_mode;
}

void mannualControl_high(const geometry_msgs::Twist msg)
{
    geometry_msgs::Twist cmd = msg;
    /*pose.pose.position.z+=cmd.linear.x/50;
    pose.pose.orientation.z+=cmd.angular.z/500;
    if(pose.pose.orientation.z > 1)
        pose.pose.orientation.z=1;
    else if(pose.pose.orientation.z < -1)
        pose.pose.orientation.z=-1;*/

    setVelocity.twist.linear.z = cmd.linear.x;
    if(cmd.angular.z > 0)
    {
        setVelocity.twist.angular.z = cmd.angular.z;//0.5;
    }
    else if(cmd.angular.z < 0)
    {
        setVelocity.twist.angular.z = cmd.angular.z;//-0.5;
    }
    else
    {
        setVelocity.twist.angular.z = 0;
    }
    

    flight_mode = velocity_mode;
}

void target_tracking(const geometry_msgs::Twist msg)
{
    geometry_msgs::Twist cmd = msg;
    /*if(cmd.linear.y > 1.5)
    {
        pose.pose.position.y -= 0.1;
    }
    else if(cmd.linear.y < 0.5)
    {
        pose.pose.position.y += 0.1;
    }*/

    //check object in center

        //check object in center
    if(cmd.linear.x > ((image_width/2)+20))
    {
        setVelocity.twist.angular.z = 0.5;
    }
    else if(cmd.linear.x < ((image_width/2)-20))
    {
        setVelocity.twist.angular.z = -0.5;
    }
    else
    {
        setVelocity.twist.angular.z = 0;
    }
}



int main(int argc, char **argv)
{
    ros::init(argc, argv, "offb_node");
    ros::NodeHandle nh;

    ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>
            ("mavros/state", 10, state_cb);
    ros::Publisher local_pos_pub = nh.advertise<geometry_msgs::PoseStamped>
            ("mavros/setpoint_position/local", 10);
    ros::Publisher velocity_pub = nh.advertise<geometry_msgs::TwistStamped>
            ("/mavros/setpoint_velocity/cmd_vel", 10);
    ros::ServiceClient arming_client = nh.serviceClient<mavros_msgs::CommandBool>
            ("mavros/cmd/arming");
    ros::ServiceClient set_mode_client = nh.serviceClient<mavros_msgs::SetMode>
            ("mavros/set_mode");

    ros::Subscriber target_coord_sub = nh.subscribe<geometry_msgs::PoseStamped>("target/coordinates", 10, setPoint);

    ros::Subscriber current_pose_sub = nh.subscribe<geometry_msgs::PoseStamped>("target/coordinates", 10, getCurrentPose);

    nh.getParam("image_size_width", image_width);

    //the setpoint publishing rate MUST be faster than 2Hz
    ros::Rate rate(30.0);

    flight_mode = position_mode;
    
    // wait for FCU connection
    while(ros::ok() && !current_state.connected){
        ros::spinOnce();
        rate.sleep();
    }
    
    pose.pose.position.x = 0;
    pose.pose.position.y = 0;
    pose.pose.position.z = 2;

    //send a few setpoints before starting
    for(int i = 100; ros::ok() && i > 0; --i){
        local_pos_pub.publish(pose);
        ros::spinOnce();
        rate.sleep();
    }

    mavros_msgs::SetMode offb_set_mode;
    offb_set_mode.request.custom_mode = "OFFBOARD";

    mavros_msgs::CommandBool arm_cmd;
    arm_cmd.request.value = true;

    ros::Time last_request = ros::Time::now();

    ros::Subscriber control_sub = nh.subscribe<geometry_msgs::Twist>("/cmd_vel", 10, mannualControl);
    ros::Subscriber control_high_sub = nh.subscribe<geometry_msgs::Twist>("/cmd_vel_high", 10, mannualControl_high);

    ros::Subscriber target_sub = nh.subscribe<geometry_msgs::Twist>("/human_tracking/mask_detection/target", 10, target_tracking);
    while(ros::ok()){
        if( current_state.mode != "OFFBOARD" && (ros::Time::now() - last_request > ros::Duration(5.0))){
            if( set_mode_client.call(offb_set_mode) && offb_set_mode.response.mode_sent){
                ROS_INFO("Offboard enabled");
            } else {
                ROS_INFO("Offboard enable failed!");
            }
            last_request = ros::Time::now();
        } else {
            if( !current_state.armed && (ros::Time::now() - last_request > ros::Duration(5.0))){
                if( arming_client.call(arm_cmd) && arm_cmd.response.success){
                    ROS_INFO("Vehicle armed");
                } else {
                    ROS_INFO("Vehicle not armed");
                }
                last_request = ros::Time::now();
            }
        }

        switch (flight_mode)
        {
        case position_mode:
            local_pos_pub.publish(pose);
            break;

        case velocity_mode:
            velocity_pub.publish(setVelocity);
            break;
        
        default:
            break;
        }
        

        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}