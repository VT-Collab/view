import pybullet as p
import pybullet_data
import os
import numpy as np

class Panda():

	def __init__(self, basePosition=[0, 0, 0]):
		self.urdfRootPath = pybullet_data.getDataPath()
		self.panda = p.loadURDF(os.path.join(self.urdfRootPath, "franka_panda/panda.urdf"), useFixedBase=True,
		                        basePosition=basePosition)

	# has two modes: joint space control (0) and ee-space control (1)
	# djoint is a 7-dimensional vector of joint velocities
	# dposition is a 3-dimensional vector of end-effector linear velocities
	# dquaternion is a 4-dimensional vector of end-effector quaternion velocities
	def step(self, mode=1, djoint=[0] * 7, dposition=[0] * 3, dquaternion=[0] * 4, grasp_open=True):

		self._position_control(mode=mode, djoint=djoint, dposition=dposition, dquaternion=dquaternion,
						 grasp_open=grasp_open)
		# # velocity control
		# self._velocity_control(mode=mode, djoint=djoint, dposition=dposition, dquaternion=dquaternion,
		#                        grasp_open=grasp_open)

		# update robot state measurement
		self._read_state()

	def place_ee(self, ee_position=[0.5, 0, 0.5], ee_quaternion=[1, 0, 0, 0]):
		q = self._inverse_kinematics(ee_position, ee_quaternion)
		self._reset_robot(q)

	def place_joint(self, joint_position, open_gripper=True):
		if open_gripper:
			q = list(joint_position) + [0.0, 0.0, 0.05, 0.05]
		else:
			q = list(joint_position) + [0.0, 0.0, 0.0, 0.0]
		self._reset_robot(q)

	def reset(self, q=[0.0, 0.0, 0.0, -2 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4, 0.0, 0.0, 0.05, 0.05]):
		self._reset_robot(q)

	def reset_gripper(self):
		for idx in range(9,11):
			p.resetJointState(self.panda, idx, 0.05)
		self._read_state()

	def _read_state(self):
		joint_position = [0] * 9
		joint_velocity = [0] * 9
		joint_torque = [0] * 9
		joint_states = p.getJointStates(self.panda, range(9))
		for idx in range(9):
			joint_position[idx] = joint_states[idx][0]
			joint_velocity[idx] = joint_states[idx][1]
			joint_torque[idx] = joint_states[idx][3]
		ee_states = p.getLinkState(self.panda, 11)
		ee_position = list(ee_states[4])
		ee_quaternion = list(ee_states[5])
		gripper_contact = p.getContactPoints(bodyA=self.panda, linkIndexA=10)
		self.state['joint_position'] = np.asarray(joint_position)
		self.state['joint_velocity'] = np.asarray(joint_velocity)
		self.state['joint_torque'] = np.asarray(joint_torque)
		self.state['ee_position'] = np.asarray(ee_position)
		self.state['ee_quaternion'] = np.asarray(ee_quaternion)
		self.state['ee_euler'] = np.asarray(p.getEulerFromQuaternion(ee_quaternion))
		self.state['gripper_contact'] = len(gripper_contact) > 0

	#############################################################################
	# print(np.array(joint_position).round(3))
	#############################################################################

	def _read_jacobian(self):
		linear_jacobian, angular_jacobian = p.calculateJacobian(self.panda, 11, [0, 0, 0],
		                                                        list(self.state['joint_position']), [0] * 9, [0] * 9)
		linear_jacobian = np.asarray(linear_jacobian)[:, :7]
		angular_jacobian = np.asarray(angular_jacobian)[:, :7]
		full_jacobian = np.zeros((6, 7))
		full_jacobian[0:3, :] = linear_jacobian
		full_jacobian[3:6, :] = angular_jacobian
		self.jacobian['full_jacobian'] = full_jacobian
		self.jacobian['linear_jacobian'] = linear_jacobian
		self.jacobian['angular_jacobian'] = angular_jacobian

	def _reset_robot(self, joint_position):
		self.state = {}
		self.desired = {}
		for idx in range(len(joint_position)):
			p.resetJointState(self.panda, idx, joint_position[idx])
		self._read_state()
		self.desired['joint_position'] = self.state['joint_position']
		self.desired['ee_position'] = self.state['ee_position']
		self.desired['ee_quaternion'] = self.state['ee_quaternion']

	def _inverse_kinematics(self, ee_position, ee_quaternion):
		return p.calculateInverseKinematics(self.panda, 11, list(ee_position), list(ee_quaternion))
	
	def _forward_kinematics(self, joint_position):
		# move robot to that position
		self._reset_robot(joint_position)
		return self.state['ee_position'], self.state['ee_quaternion']
	
	def _position_control(self, mode, djoint, dposition, dquaternion, grasp_open):
		if mode:
			djoint = self._inverse_kinematics(dposition, dquaternion)
		gripper_position = [0.0, 0.0]
		if grasp_open:
			gripper_position = [0.05, 0.05]
		djoint = np.hstack((djoint, gripper_position))
		p.setJointMotorControlArray(self.panda, range(11), p.POSITION_CONTROL, targetPositions=list(djoint))

	def _velocity_control(self, mode, djoint, dposition, dquaternion, grasp_open):
		if mode:
			self.desired['ee_position'] += np.asarray(dposition) / 240.0
			self.desired['ee_quaternion'] += np.asarray(dquaternion) / 240.0
			q_dot = self._inverse_kinematics(self.desired['ee_position'], self.desired['ee_quaternion']) - self.state[
				'joint_position']
		else:
			self.desired['joint_position'] += np.asarray(list(djoint) + [0, 0]) / 240.0
			q_dot = self.desired['joint_position'] - self.state['joint_position']
		gripper_position = [0.0, 0.0]
		if grasp_open:
			gripper_position = [0.05, 0.05]
		p.setJointMotorControlArray(self.panda, range(9), p.VELOCITY_CONTROL, targetVelocities=list(q_dot))
		p.setJointMotorControlArray(self.panda, [9, 10], p.POSITION_CONTROL, targetPositions=gripper_position)
