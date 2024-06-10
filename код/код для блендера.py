import bpy
import numpy as np
import math as m
import random
import sys
sys.path.append("C:\\Users\\79538\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.8_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python38\\site-packages\\cv2")
import cv2

## Main Class
class Render:
    def __init__(self):
        ## Информация о сцене
        # Определить информацию о сцене
        self.scene = bpy.data.scenes['Scene']
        # Определите информацию, относящуюся к <bpy.data.objects>
        self.camera = bpy.data.objects['Camera']
        self.light = bpy.data.objects['Light']
        self.obj_names = ['neck', 'eye_l', 'eye_r', 'skin', 'mouth', 'mouth_2', 'mouth_3', 'hair', 'eye_lashes1', 'eye_lashes2', 'eye_l_1', 'eye_r_1']
        self.face = bpy.data.objects['skin']
        self.eye_l = bpy.data.objects['eye_l']
        self.eye_r = bpy.data.objects['eye_r']
        self.eye_l_1 = bpy.data.objects['eye_l_1']
        self.eye_r_1 = bpy.data.objects['eye_r_1']
        self.objects = self.create_objects()  # Создать список bpy.data.objects из bpy.data.objects[1] в bpy.data.objects[N]
                
        # Выберем основной объект
        main_object = bpy.data.objects['skin']# Выберем объект, который хотим привязать к основному
        child_object1 = bpy.data.objects['eye_l']
        child_object2 = bpy.data.objects['eye_r']
        child_object3 = bpy.data.objects['mouth_2']
        child_object4 = bpy.data.objects['mouth_3']
        child_object5 = bpy.data.objects['eye_lashes1']
        child_object6 = bpy.data.objects['eye_lashes2']
        child_object7 = bpy.data.objects['mouth']
        child_object8 = bpy.data.objects['hair']
        child_object9 = bpy.data.objects['eye_l_1']
        child_object10 = bpy.data.objects['eye_r_1']
        # Добавим констрейнт "Child Of" к child_object
        constraint1 = child_object1.constraints.new('CHILD_OF')
        constraint1.target = main_object
        constraint1.inverse_matrix = main_object.matrix_world.inverted()
        # Применяем констрейнт

        
        constraint2 = child_object2.constraints.new('CHILD_OF')
        constraint2.target = main_object
        constraint2.inverse_matrix = main_object.matrix_world.inverted()

        
        constraint3 = child_object3.constraints.new('CHILD_OF')
        constraint3.target = main_object
        constraint3.inverse_matrix = main_object.matrix_world.inverted()

        
        constraint4 = child_object4.constraints.new('CHILD_OF')
        constraint4.target = main_object
        constraint4.inverse_matrix = main_object.matrix_world.inverted()
        
        constraint5 = child_object5.constraints.new('CHILD_OF')
        constraint5.target = main_object
        constraint5.inverse_matrix = main_object.matrix_world.inverted()
        
        constraint6 = child_object6.constraints.new('CHILD_OF')
        constraint6.target = main_object
        constraint6.inverse_matrix = main_object.matrix_world.inverted()
        
        constraint7 = child_object7.constraints.new('CHILD_OF')
        constraint7.target = main_object
        constraint7.inverse_matrix = main_object.matrix_world.inverted()
        
        constraint8 = child_object8.constraints.new('CHILD_OF')
        constraint8.target = main_object
        constraint8.inverse_matrix = main_object.matrix_world.inverted()
        
        constraint9 = child_object9.constraints.new('CHILD_OF')
        constraint9.target = main_object
        constraint9.inverse_matrix = main_object.matrix_world.inverted()
        
        constraint10 = child_object10.constraints.new('CHILD_OF')
        constraint10.target = main_object
        constraint10.inverse_matrix = main_object.matrix_world.inverted()
        bpy.context.view_layer.update()
        # Теперь child_object будет следовать за main_object во всех перемещениях и трансформациях
        
        
        ## Рендеринг информации
        self.light_d_limits = [0, 2]  # Определите диапазон высот z в метрах, через который будет проходить камера.
        self.beta_limits = [80, -80]  # Определите диапазон бета-углов, через который будет панорамироваться камера.
        self.gamma_limits = [0, 360]  # Определите диапазон гамма-углов, через который будет панорамироваться камера.

        ## Вывод информации
        # Введите предпочитаемое вами местоположение для изображений и меток.
        self.images_filepath = 'D:\блендер\чел'
        self.labels_filepath = 'D:\блендер\чел'
        
        

    def set_eyes(self):

        angle_x = m.radians(random.randint(-15, 15))
        angle_z = m.radians(random.randint(330, 390))

        self.eye_l.rotation_euler.x = angle_x
        self.eye_l.rotation_euler.z = angle_z
        self.eye_l.rotation_euler.y = m.radians(0)

        self.eye_r.rotation_euler.x = angle_x
        self.eye_r.rotation_euler.z = angle_z
        self.eye_r.rotation_euler.y = m.radians(0)

    def set_light(self):
        self.light.location = (2, -400, 2)

    def get_all_coordinates(self):
        '''
        Эта функция не принимает никаких входных данных и выводит полную строку с координатами
        всех объектов, видимых на текущем изображении
        '''
        main_text_coordinates = ''  # Инициализируем переменную, в которой мы будем хранить координаты.
        for i, object in enumerate(self.objects):  # Перебрать все объекты
            print("     On object:", object)
            b_box = self.find_bounding_box(object)  #Получить координаты текущего объекта
            if b_box:  # Если find_bounding_box() не возвращает None
                print("         Initial coordinates:", b_box)
                text_coordinates = self.format_coordinates(b_box, i)  # Переформатировать координаты в формат YOLOv3.
                print("         YOLO-friendly coordinates:", text_coordinates)
                main_text_coordinates = main_text_coordinates + text_coordinates  # Обновите переменные координат основного текста, указав каждую
                # строку, соответствующую каждому классу в кадре текущего изображения.
            else:
                print("         Object not visible")
                pass

        return main_text_coordinates  # Вернуть все координаты
    def format_coordinates(self, coordinates, classe):
        '''
        Эта функция принимает в качестве входных данных координаты, созданные функцией find_bounding box(), текущий класс,
        ширину изображения и высоту изображения и выводит координаты ограничивающей рамки текущего класса
        '''
        # Если текущий класс находится в поле зрения камеры
        if coordinates:
            ## Изменить систему отсчета координат
            x1 = (coordinates[0][0])
            x2 = (coordinates[1][0])
            y1 = (1 - coordinates[1][1])
            y2 = (1 - coordinates[0][1])

            ## Получите окончательную информацию об ограничивающей рамке
            width = (x2-x1)  # Вычислить абсолютную ширину ограничивающей рамки
            height = (y2-y1) # Вычислить абсолютную высоту ограничивающей рамки
            # Вычислить абсолютную высоту ограничивающей рамки
            cx = x1 + (width/2)
            cy = y1 + (height/2)

            ## Сформулируйте линию, соответствующую ограничивающему прямоугольнику одного класса
            txt_coordinates = str(classe) + ' ' + str(cx) + ' ' + str(cy) + ' ' + str(width) + ' ' + str(height) + '\n'

            return txt_coordinates
        # Если текущий класс не виден камере, передайте
        else:
            pass

    def find_bounding_box(self, obj):
        """
        Возвращает ограничивающую рамку пространства камеры объекта-сетки.

        Получает ограничивающую рамку кадра камеры, которая по умолчанию возвращается без применения каких-либо преобразований.
        Создайте новый объект-сетку на основе self.carre_bleu и отмените все преобразования, чтобы он находился в том же пространстве, что и объект-сетка.
        рамка камеры. Найдите минимальные/максимальные координаты вершин сетки, видимой в кадре, или «Нет», если сетка не видна.

        :param scene:
        :param camera_object:
        :param mesh_object:
        :return:
        """

        """ Получите матрицу обратного преобразования. """
        matrix = self.camera.matrix_world.normalized().inverted()
        """ Создайте новый блок данных сетки, используя матрицу обратного преобразования, чтобы отменить любые преобразования."""
        mesh = obj.to_mesh(preserve_all_data_layers=True)
        mesh.transform(obj.matrix_world)
        mesh.transform(matrix)

        """ Получите мировые координаты ограничивающей рамки кадра камеры перед любыми преобразованиями. """
        frame = [-v for v in self.camera.data.view_frame(scene=self.scene)[:3]]

        lx = []
        ly = []

        for v in mesh.vertices:
            co_local = v.co
            z = -co_local.z

            if z <= 0.0:
                """ Вертекс находится за камерой; игнорируй это. """
                continue
            else:
                """ Перспективное деление"""
                frame = [(v / (v.z / z)) for v in frame]

            min_x, max_x = frame[1].x, frame[2].x
            min_y, max_y = frame[0].y, frame[1].y

            x = (co_local.x - min_x) / (max_x - min_x)
            y = (co_local.y - min_y) / (max_y - min_y)

            lx.append(x)
            ly.append(y)

        """ Изображение не отображается, если все вершины сетки были проигнорированы. """
        if not lx or not ly:
            return None

        min_x = np.clip(min(lx), 0.0, 1.0)
        min_y = np.clip(min(ly), 0.0, 1.0)
        max_x = np.clip(max(lx), 0.0, 1.0)
        max_y = np.clip(max(ly), 0.0, 1.0)

        """ Изображение не отображается, если обе ограничивающие точки находятся на одной стороне. """
        if min_x == max_x or min_y == max_y:
            return None

        """ Определите размер визуализированного изображения """
        render = self.scene.render
        fac = render.resolution_percentage * 0.01
        dim_x = render.resolution_x * fac
        dim_y = render.resolution_y * fac

        ## Убедитесь, что нет координат, равных нулю.
        coord_list = [min_x, min_y, max_x, max_y]
        if min(coord_list) == 0.0:
            indexmin = coord_list.index(min(coord_list))
            coord_list[indexmin] = coord_list[indexmin] + 0.0000001

        return (min_x, min_y), (max_x, max_y)
    

    def rotate_points(self, points, angles):
        """
        Rotate 3D points on angles.

        Rotation sequence: OZ(roll), OY(yaw), OX(pitch)

        Arguments:
            - points (np.array[:, 3]):
                keypoints [[x1, y1, z1], ...]
            - angles (np.array[3]):
                angles [pitch, yaw, roll], in degrees

        Returns:
            rotated points, np.array[:, 3]
        """
        (pitch, yaw, roll) = np.radians(angles)

        xr = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), np.sin(pitch)],
            [0, -np.sin(pitch), np.cos(pitch)]
        ])
        yr = np.array([
            [np.cos(yaw), 0, -np.sin(yaw)],
            [0, 1, 0],
            [np.sin(yaw), 0, np.cos(yaw)]
        ])
        zr = np.array([
            [np.cos(roll), np.sin(roll), 0],
            [-np.sin(roll), np.cos(roll), 0],
            [0, 0, 1]
        ])

        return points @ (zr@yr@xr)

    def rotmat_to_angles(self, m):
        if abs(m[2, 0] + 1) <= 1e-9:
            return np.degrees([
                -np.arctan2(m[0, 1], m[0, 2]),
                -np.pi/2,
                0
            ])

        if abs(m[2, 0] - 1) <= 1e-9:
            return np.degrees([
                -np.arctan2(-m[0, 1], -m[0, 2]),
                np.pi/2,
                0
            ])

        return np.degrees([
            -np.arctan2(m[2, 1], m[2, 2]),
            np.arcsin(m[2, 0]),
            -np.arctan2(m[1, 0], m[0, 0])
        ])
    
    def rotate_angles(self, angles, pitch, yaw, roll):
        """
        Rotate angles on angle.

        Arguments:
            - angles (np.array[3]):
                angles [pitch, yaw, roll], in degrees
            - pitch, yaw, roll (float):
                angles, in degrees

        Returns:
            rotated angles, np.array[3], in degrees
        """
        
        
        rotated_points = self.rotate_points(np.eye(3), angles)
        return self.rotmat_to_angles(self.rotate_points(rotated_points, [pitch, yaw, roll])).astype(np.float32)

    def draw_angles(self, image, line_size=40, thickness=2):
        "angles: (pitch, yaw, roll)"
        RED = (0, 0, 255)
        GREEN = (0, 255, 0)
        BLUE = (255, 0, 0)
        
        rotation = self.face.rotation_euler
        rotation_degrees = [m.degrees(r) for r in rotation]
        (pitch, yaw, roll) = np.array(rotation_degrees)
#        pitch = -20
#        yaw = -10
#        roll = -20
        roll = -roll
        
#        angles = [m.degrees(r) for r in rotation]
        angles = [4.65, 3.79, -3.13]
        angles = np.array(angles)
                
        rotated_angles = self.rotate_angles(angles, pitch, yaw, roll)

        rot_points = self.rotate_points(np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, -1]
    ]), np.array(rotated_angles))[:, :2]
        
        
        pos = np.array([100,100], dtype=int)
        axis = (line_size*rot_points).astype(int)

        
        for (point, color) in zip(axis, [RED, GREEN, BLUE]):
            image = cv2.line(image, tuple(pos), tuple(pos + point), color, thickness)
            

        return image

    def main_rendering_loop(self, rot_step):
        '''
        Эта функция представляет собой основной алгоритм, описанный в руководстве. Она принимает
        шаг вращения в качестве входных данных и выводит изображения и метки в указанные выше места.
        '''
        ## Рассчитайте количество изображений и меток для создания
        n_renders = self.calculate_n_renders(rot_step)  # Рассчитать количество изображений
        print('Number of renders to create:', n_renders)
    
        # accept_render = input('\nContinue?[Y/N]:  ') # Спросите, продолжать ли генерацию данных
        accept_render = 'Y'
        
        
        a = 1
        if accept_render == 'Y':  # Если пользователь вводит «Y», приступайте к генерации данных.
                    # Создайте файл .txt, в котором записывается ход генерации данных.
            report_file_path = self.labels_filepath + 'progress_report.json'
            report = open(report_file_path, 'w')
                    # Умножьте пределы на 10, чтобы адаптироваться к циклу for.
            dmin = int(self.light_d_limits[0] * 10)
            dmax = int(self.light_d_limits[1] * 10)
                    # Определите счетчик для именования каждого выводимого файла .png и .txt.
            render_counter = 0
                    # Определите шаг, с которым будут делаться снимки.
            rotation_step = rot_step
            
            
            # Начало вложенных циклов
            
            while render_counter == 0:
                for d in range(dmin, dmax + 1, 2):  # Петля для изменения высоты света
                    ## Обновить высоту света
                    self.light.location = (0, -1, d / 10)  # Разделите расстояние z на 10, чтобы пересчитать текущую высоту.

                    # Рефакторинг бета-пределов, чтобы они находились в диапазоне от 0 до 360, чтобы адаптировать ограничения к циклу for.
                    min_beta = (-1) * self.beta_limits[0] + 90
                    max_beta = (-1) * self.beta_limits[1] + 90
                   

                    for beta in range(min_beta, max_beta + 1, rotation_step):  # Петля для изменения угла бета
                        beta_r = (-1) * beta + 90  # Перефакторить текущую бета-версию

                        for gamma in range(self.gamma_limits[0], self.gamma_limits[1] + 1,
                                           rotation_step):  # Петля для изменения угла гаммы
                                               
                            for angle_x in range(-7, 10, 9):
                                self.eye_l.rotation_euler.x = m.radians(angle_x)
                                self.eye_r.rotation_euler.x = m.radians(angle_x)
                                self.eye_l_1.rotation_euler.x = m.radians(angle_x)
                                self.eye_r_1.rotation_euler.x = m.radians(angle_x)

                                for angle_z in range(345, 375, 15):
                                    self.eye_l.rotation_euler.z = m.radians(angle_z)
                                    self.eye_r.rotation_euler.z = m.radians(angle_z)
                                    self.eye_l_1.rotation_euler.z = m.radians(angle_z)
                                    self.eye_r_1.rotation_euler.z = m.radians(angle_z)
                             
                                    for x in range(-20, 20, 15):
                                        self.face.rotation_euler.x = m.radians(x)
                                        
                                        for y in range(-10, 10, 7):
                                            self.face.rotation_euler.y = m.radians(y)
                                            
                                            for z in range(-20, 20, 15):
                                                self.face.rotation_euler.z = m.radians(z)
             
                                                render_counter += 1  # Обновить счетчик
                                                
                                                ## Обновить вращение оси
                                                axis_rotation = (m.radians(beta_r), 0, m.radians(gamma))
                                                # self.axis.rotation_euler = axis_rotation # Назначьте вращение <bpy.data.objects['Empty']> object
                                                # Отображение демонстрационной информации — Расположение камеры
                                                print("On render:", render_counter)
                                                print("--> Location of the ligth:")
                                                print("     d:", d / 10, "m")
                                                print("     Beta:", str(beta_r) + "Degrees")
                                                print("     Gamma:", str(gamma) + "Degrees")

                                                ## Configure lighting можно поставить не рандом а например цикл
                                                energy = random.randint(0, 30)  # Захват случайной интенсивности света
                                                self.light.data.energy = energy  # Обновите <bpy.data.objects['Light']> energy information

                                                ## Создать рендер
                                                #self.render_blender(render_counter)
                                                image_path = self.render_blender(render_counter)
                                                image = cv2.imread(image_path)  
                                                
                                                image = self.draw_angles(image)
                                                cv2.imwrite(f"D:\блендер\чел\{render_counter}.png", image)
                                                  # Сфотографируйте текущую сцену и выведите файл render counter.png
                                                # Отображение демонстрационной информации — Информация о фотографии
                                                print("--> Picture information:")
                                                print("     Resolution:", (self.xpix * self.percentage, self.ypix * self.percentage))
                                                print("     Rendering samples:", self.samples)

                                                ## Output Labels
                                                text_file_name = self.labels_filepath + '/' + str(
                                                    render_counter) + '.json'  # Создать имя файла этикетки
                                                text_file = open(text_file_name, 'w+')  # Открыть .txt-файл этикетки
                                                 # Получить отформатированные координаты ограничивающих рамок всех объектов сцены
                                                 # Отображение демонстрационной информации - Создание этикетки
                                                print("---> Label Construction")
                                                text_coordinates = self.get_all_coordinates()
                                                splitted_coordinates = text_coordinates.split('\n')[:-1]  # Удалить последний '\n' в координатах
                                                text_file.write('\n'.join(splitted_coordinates))  # Запишите координаты в текстовый файл и выведите файл render_counter.txt.
                                                text_file.close()  # Закройте файл .txt, соответствующий метке.
                                                ## Показать ход выполнения пакета рендеринга
                                                print('Progress =', str(render_counter) + '/' + str(n_renders))
                                                report.write('Progress: ' + str(render_counter) + ' Rotation: ' + str(axis_rotation) + ' z_d: ' + str(d / 10) + '\n')
                                                
                                                
                                                if a > 5:
                                                    exit()
                                                a += 1
                                
                report.close()  # Закройте файл .txt, соответствующий отчету.
                
            

    def render_blender(self, count_f_name):
        # Определить случайные параметры
        random.seed(random.randint(1, 10))
        self.xpix = 1000
        self.ypix = 1000
        self.percentage = 100
        self.samples = 15
        # Рендеринг изображений
        image_name = str(count_f_name) + '.png'
        self.export_render(self.xpix, self.ypix, self.percentage, self.samples, self.images_filepath, image_name)
        return f"D:\блендер\чел\{image_name}"
        
    def export_render(self, res_x, res_y, res_per, samples, file_path, file_name):
        # Установите все параметры сцены
        bpy.context.scene.cycles.samples = samples
        self.scene.render.resolution_x = res_x
        self.scene.render.resolution_y = res_y
        self.scene.render.resolution_percentage = res_per
        self.scene.render.filepath = file_path + '/' + file_name

        #Сфотографировать текущую видимую сцену
        bpy.ops.render.render(write_still=True)

    def calculate_n_renders(self, rotation_step):
        zmin = int(self.light_d_limits[0] * 10)
        zmax = int(self.light_d_limits[1] * 10)

        render_counter = 0
        rotation_step = rotation_step

        for d in range(zmin, zmax + 1, 2):
            light_location = (0, 0, d / 10)
            min_beta = (-1) * self.beta_limits[0] + 90
            max_beta = (-1) * self.beta_limits[1] + 90

            for beta in range(min_beta, max_beta + 1, rotation_step):
                beta_r = 90 - beta

                for gamma in range(self.gamma_limits[0], self.gamma_limits[1] + 1, rotation_step):
                    render_counter += 1
                    axis_rotation = (beta_r, 0, gamma)

        return render_counter

    def create_objects(self):  # Эта функция создает список всех <bpy.data.objects>.
        objs = []
        for obj in self.obj_names:
            objs.append(bpy.data.objects[obj])

        return objs
    
    def pr(self):
        "angles: (pitch, yaw, roll)"
        RED = (0, 0, 255)
        GREEN = (0, 255, 0)
        BLUE = (255, 0, 0)
        
        rotation = self.face.rotation_euler
        rotation_degrees = [m.degrees(r) for r in rotation]
        (pitch, yaw, roll) = np.array(rotation_degrees)
            

        print(rotation)
        print(rotation_degrees)
        print(yaw)


## Запустить генерацию данных
if __name__ == '__main__':
    # Инициализировать класс рендеринга как r
    r = Render()
    # Инициализировать свет
    r.set_light()
#    r.set_eyes()
    # Начать генерацию данных
    rotation_step = 1000000  # (360/x)*(160/x)*4
    r.main_rendering_loop(rotation_step)
#    r.pr()
