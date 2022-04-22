#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/Twist.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <std_msgs/String.h>
#include <iostream>

using namespace std;


mavros_msgs::State current_state;
geometry_msgs::PoseStamped pose;
geometry_msgs::PoseStamped current_pose;
geometry_msgs::TwistStamped setVelocity;

//task one is for uav drawing circle to find target
//setting circle radius to 1 meter
double task_one_radius = 1;
double pi = 3.14;

//set task one position
geometry_msgs::PoseStamped task_one_set_point;

//camera resolution
int image_width;
int image_height;

//uav found target when task one
bool tracking = false;

//flight mode
enum Flight_Mode {position_mode, velocity_mode, manual_control, armed};
Flight_Mode flight_mode;

//get mavros state
void state_cb(const mavros_msgs::State::ConstPtr& msg)
{
    current_state = *msg;
}

//set next point for uav
void setPoint(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    geometry_msgs::PoseStamped set_point = *msg;
    pose.pose = set_point.pose;
}

//get current position from vicon
void getCurrentPose(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    current_pose = *msg;
}

//for mannual control, linear: forward, backward, left and right
void mannualControl(const geometry_msgs::Twist msg)
{
    geometry_msgs::Twist cmd = msg;
    if(flight_mode==manual_control)
    {
        setVelocity.twist.linear.x = cmd.linear.x;
        setVelocity.twist.linear.y = cmd.angular.z;
    }
}

//for mannual control, linear: high, low. angular: yaw
void mannualControl_high(const geometry_msgs::Twist msg)
{
    geometry_msgs::Twist cmd = msg;
    if(flight_mode==manual_control)
    {
        setVelocity.twist.linear.z = cmd.linear.x;
        setVelocity.twist.angular.z = cmd.angular.z;
        setVelocity.twist.angular.z = cmd.angular.z;
    }
}

//counte the time that uav resume tracking task to search new target
int counter = 0;

void target_tracking(const geometry_msgs::Twist msg)
{
    geometry_msgs::Twist cmd = msg;
    if(cmd.linear.x ==0 &&cmd.linear.y==0)
    {
        setVelocity.twist.angular.z = 0;
        setVelocity.twist.angular.x = 0;
        ROS_INFO("no target");
        counter++;
        if(counter>=100)
        {
            tracking = false;
        }
    }
    else
    {
        if(cmd.linear.x > ((image_width/2)+30))
        {
            setVelocity.twist.angular.z = -0.3;
            ROS_INFO("Target tracking r");
        }
        else if(cmd.linear.x < ((image_width/2)-30))
        {
            setVelocity.twist.angular.z = 0.3;
            ROS_INFO("Target tracking l");
        }
        else
        {
            setVelocity.twist.angular.z = 0;
            ROS_INFO("Target tracking c");
        }

        /*if(cmd.linear.y > ((image_height/2)+30))
        {

            setVelocity.twist.angular.x = 0.1;
            ROS_INFO("Target tracking, too far");
        }
        else if(cmd.linear.y < ((image_height/2)-30))
        {

            setVelocity.twist.angular.x = -0.1;
            ROS_INFO("Target tracking l, too close");
        }
        else
        {
            setVelocity.twist.angular.x = 0;
            ROS_INFO("Target tracking, keep distance");
        }*/

        tracking = true;
        flight_mode = velocity_mode;
        counter=0;
    }

    //keep high position in velocity control
    if(current_pose.pose.position.z<2.5)
    {
        setVelocity.twist.linear.z = 0.1;
    }
    else
    {
        setVelocity.twist.linear.z = 0;
    }
}

//get mode from user control GUI
void getMode(const std_msgs::String mode)
{
    if(mode.data=="armed")
    {
        flight_mode=armed;
    }
    else if(mode.data=="task1")
    {
        flight_mode=velocity_mode;
    }
}

//Find the shortest distance to resume task 1
double shortest_distance(double r)
{
    //current position x and y
    double current_x = current_pose.pose.position.x;
    double current_y = current_pose.pose.position.y;

    double min_distance = 9999;
    double min_distance_angle = 0;
    double distance = 0;

    //fing point in all angle of a circle
    for(int i=1;i<=360;i++)
    {
        double test_x = r*cos(i*(pi/180));
        double test_y = r*sin(i*(pi/180));
        distance = sqrt(pow(current_x - test_x, 2) + pow(current_y - test_y, 2) * 1.0);
        if(distance<=min_distance)
        {
            min_distance=distance;
            min_distance_angle = i;
        }
    }
    return min_distance_angle;
}


int main(int argc, char **argv)
{
    //rosnode name
    ros::init(argc, argv, "offb_node");

    //handler for rosnode
    ros::NodeHandle nh;

    //get mavros state
    ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>("mavros/state", 10, state_cb);

    //publish the point for uav to flight, used for position control
    ros::Publisher local_pos_pub = nh.advertise<geometry_msgs::PoseStamped>("mavros/setpoint_position/local", 10);

    //publish velocity command to mavros, used for velocity control
    ros::Publisher velocity_pub = nh.advertise<geometry_msgs::TwistStamped>("/mavros/setpoint_velocity/cmd_vel", 10);

    //check uav arm state
    ros::ServiceClient arming_client = nh.serviceClient<mavros_msgs::CommandBool>("mavros/cmd/arming");

    //set mavros flight mode
    ros::ServiceClient set_mode_client = nh.serviceClient<mavros_msgs::SetMode>("mavros/set_mode");

    //get target position   
    ros::Subscriber target_coord_sub = nh.subscribe<geometry_msgs::PoseStamped>("/user_control/setPoint_pose", 10, setPoint);

    //get current position of uav
    ros::Subscriber current_pose_sub = nh.subscribe<geometry_msgs::PoseStamped>("/mavros/local_position/pose", 10, getCurrentPose);

    //get camera resolution
    nh.param("image_size_width", image_width, 640);// image_width=1280; //realsense cam: 1280, E2ES: 640
    nh.param("image_size_height", image_height, 480);

    //the setpoint publishing rate MUST be faster than 2Hz
    ros::Rate rate(30.0);

    //default flight mode: position mode
    flight_mode = position_mode;

    //store task one values
    double task_one_current_angle = 0;
    double self_rotate_angle = 0;

    ROS_INFO("wait for FCU connection"); 
    while(ros::ok() && !current_state.connected){
        ros::spinOnce();
        rate.sleep();
    }
    
    //take off
    pose.pose.position.x = 0;
    pose.pose.position.y = 0;
    pose.pose.position.z = 2;

    ROS_INFO("send a few setpoints before starting"); 
    for(int i = 100; ros::ok() && i > 0; --i){
        local_pos_pub.publish(pose);
        ros::spinOnce();
        rate.sleep();
    }

    //flight high level
    pose.pose.position.z = 2.5;
    task_one_set_point.pose.position.z = 2.5;

    mavros_msgs::SetMode offb_set_mode;
    offb_set_mode.request.custom_mode = "OFFBOARD";

    mavros_msgs::CommandBool arm_cmd;
    arm_cmd.request.value = true;

    ros::Time last_request = ros::Time::now();

    //manual control command
    ros::Subscriber control_sub = nh.subscribe<geometry_msgs::Twist>("/user_control/cmd_vel", 10, mannualControl);
    ros::Subscriber control_high_sub = nh.subscribe<geometry_msgs::Twist>("/user_control/cmd_vel_high", 10, mannualControl_high);

    //get target position
    ros::Subscriber target_sub = nh.subscribe<geometry_msgs::Twist>("/human_tracking/mask_detection/target", 10, target_tracking);

    //get flight mode from user control GUI
    ros::Subscriber flight_mode_sub = nh.subscribe<std_msgs::String>("/user_control/set_mode", 1, getMode);

    //flag for uav from task one to tracking mode
    bool resume_task_one = true;

    double angle_tick = 1;

    while(ros::ok()){
        //checking mode status and set to offboard mode
        if( current_state.mode != "OFFBOARD" && (ros::Time::now() - last_request > ros::Duration(5.0))){
            if( set_mode_client.call(offb_set_mode) && offb_set_mode.response.mode_sent){
                ROS_INFO("Offboard enabled");
            } else {
                ROS_INFO("Offboard enable failed!");
            }
            last_request = ros::Time::now();
        } else {
            //check arm status and arm the uav 
            if( !current_state.armed && (ros::Time::now() - last_request > ros::Duration(5.0))){
                if( arming_client.call(arm_cmd) && arm_cmd.response.success){
                    ROS_INFO("Vehicle armed");
                } else {
                    ROS_INFO("Vehicle not armed");
                }
                last_request = ros::Time::now();
            }
        }

        //switch between position flight mode and velocity flight mode
        switch (flight_mode)
        {
            case position_mode:
                local_pos_pub.publish(pose);
                break;

            case velocity_mode:
                if(tracking==true)
                {
                    //follow target
                    velocity_pub.publish(setVelocity);
                    resume_task_one=true;
                }
                else
                {
                    //check if uav resume from tracking task
                    if(resume_task_one)
                    {
                        resume_task_one=false;
                        
                        //if uav resume from tarcking task, find the shortest point that the uav flight the circle
                        task_one_current_angle = shortest_distance(task_one_radius);
                    }
                    else
                    {
                        //get next x, y point use radius and angle of circle
                        task_one_set_point.pose.position.x = task_one_radius*cos(task_one_current_angle*(pi/180));
                        task_one_set_point.pose.position.y = task_one_radius*sin(task_one_current_angle*(pi/180));

                        //set uav direction while drawing circle. 
                        //+0: backward to center of circle
                        //+90: forward to the move direction
                        //+180: forward to center of circle (convert cos and sin)
                        //+360: backward to the move direction (convert cos and sin)
                        self_rotate_angle = task_one_current_angle + 90;
                        if(self_rotate_angle > 360)
                            self_rotate_angle -= 360;

                        //set orientation of uav
                        task_one_set_point.pose.orientation.z = sin((self_rotate_angle/2)*(pi/180));
                        task_one_set_point.pose.orientation.w = cos((self_rotate_angle/2)*(pi/180));

                        //publish position
                        local_pos_pub.publish(task_one_set_point);

                        //increase the angle, change divider to change flight speed
                        task_one_current_angle += angle_tick/double(2.8);
                        if(task_one_current_angle>360)
                            task_one_current_angle=0;
                    }
                }
                
                break;
            
            default:
                break;
        }
        
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}