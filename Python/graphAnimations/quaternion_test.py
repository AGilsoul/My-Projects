from Vector import *


def in_menu():
    user_choice = int(input('Select an Option:\n'
                            '-1.) Rotate vector around an axis using quaternions\n'
                            '-2.) Rotate vector given quaternion input\n'
                            '-3.) Convert Euler angles (degrees) to quaternion\n'
                            '-4.) Convert Euler angles (radians to quaternion\n'
                            '-5.) Convert quaternion to Euler angles (radians)\n'
                            '-6.) Quit\n'))
    if user_choice == 1:
        print('\nRotate vector around an axis using quaternions')
        p_x = float(input('Enter x-coordinate of vector to rotate: '))
        p_y = float(input('Enter y-coordinate of vector to rotate: '))
        p_z = float(input('Enter z-coordinate of vector to rotate: '))
        p_0 = np.array([p_x, p_y, p_z])
        print('Vector: ' + str(p_0))

        r_x = float(input('Enter x-coordinate of rotation axis vector: '))
        r_y = float(input('Enter y-coordinate of rotation axis vector: '))
        r_z = float(input('Enter z-coordinate of rotation axis vector: '))
        r = np.array([r_x, r_y, r_z])
        print('Rotation Axis: ' + str(r))

        radians = bool(int(input('Use degrees(0) or radians(1): ')))
        t = float(input('Enter value of theta: '))
        if radians:
            print('Using radians')
            print('Rotated vector: ' + str(Quaternion.rotate_rad(p_0, r, t, verbose=True)))
        else:
            print('using degrees')
            print('Rotated vector: ' + str(Quaternion.rotate_deg(p_0, r, t, verbose=True)))

    elif user_choice == 2:
        print('\nRotate vector given quaternion input')
        p_x = float(input('Enter x-coordinate of vector to rotate: '))
        p_y = float(input('Enter y-coordinate of vector to rotate: '))
        p_z = float(input('Enter z-coordinate of vector to rotate: '))
        p_0 = np.array([p_x, p_y, p_z])
        print('Vector: ' + str(p_0))

        q_a = float(input('Enter value of a: '))
        q_b = float(input('Enter value of b: '))
        q_c = float(input('Enter value of c: '))
        q_d = float(input('Enter value of d: '))
        q = np.array([q_a, q_b, q_c, q_d])
        print('Quaternion: ' + str(q))

        print('Rotated vector: ' + str(Quaternion.rotate(p_0, q)))

    elif user_choice == 3:
        print('\nConvert Euler angles (degrees) to quaternion')
        phi = float(input('Enter phi'))
        theta = float(input('Enter theta'))
        psi = float(input('Enter psi'))
        print('Euler angles: [{}, {}, {}]'.format(phi, theta, psi))
        print('Quaternion: ' + str(Quaternion.euler_to_quaternion_deg([phi, theta, psi])))

    elif user_choice == 4:
        print('\nConvert Euler angles (radians) to quaternion')
        phi = float(input('Enter phi'))
        theta = float(input('Enter theta'))
        psi = float(input('Enter psi'))
        print('Euler angles: [{}, {}, {}]'.format(phi, theta, psi))
        print('Quaternion: ' + Quaternion.to_string(Quaternion.euler_to_quaternion_rad([phi, theta, psi])))

    elif user_choice == 5:
        print('\nConvert quaternion to Euler angles (radians)')
        q_a = float(input('Enter value of a: '))
        q_b = float(input('Enter value of b: '))
        q_c = float(input('Enter value of c: '))
        q_d = float(input('Enter value of d: '))
        q = np.array([q_a, q_b, q_c, q_d])
        print('Quaternion: {} + {}i + {}j + {}k'.format(q_a, q_b, q_c, q_d))
        print('Euler angles: ' + str(Quaternion.quaternion_to_euler_(q)))
    else:
        return 0
    return 1


while True:
    if in_menu() == 0:
        break
