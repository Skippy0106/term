cmake_minimum_required(VERSION 3.0.2)
project(term)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)
find_package(OpenCV REQUIRED)
find_package(Eigen3)
catkin_package()


include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

add_executable(term
 
  src/term.cpp)
target_link_libraries(term ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})



