import bpy
import numpy as np
import math as m
import random
import cv2
import rpycocotools.anns
from rpycocotools import mask
import bpycv
import json
import datetime
import os
import uuid

## Main Class
class Render:
    
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    
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
        main_object = bpy.data.objects['skin']
        # Выберем объекты, которые хотим привязать к основному
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
        self.light_d_limits = [0, 2]  # Определите диапазон высот z в метрах, через который будет проходить свет.
        self.beta_limits = [80, -80]  # Определите диапазон бета-углов, через который будет панорамироваться свет.
        self.gamma_limits = [0, 360]  # Определите диапазон гамма-углов, через который будет панорамироваться свет.

        ## Вывод информации
        # Введите предпочитаемое вами местоположение для изображений и меток.
        self.images_filepath = 'D:\blender\renders' #ставите свой путь
        self.labels_filepath = 'D:\blender\renders' #ставите свой путь

    def set_light(self):
        self.light.location = (2, -400, 2)

    def get_obj_mask(self, obj):
        '''
        Получение маски объекта obj
        '''
        for i in range(len(obj)):
            bpy.data.objects[obj[i]]["inst_id"] = 100000 + i
        result = bpycv.render_data()
        segmentation = np.uint8(result["inst"])
        for i in range(len(obj)):
            bpy.data.objects[obj[i]]["inst_id"] = 0
        return segmentation
    
    def get_bbox_from_rle(self, rle):
        return mask.to_bbox(rle)

    def get_list_from_bbox_obj(self, bbox: rpycocotools.anns.BBox):
        return [bbox.left, bbox.top, bbox.width, bbox.height]
    
    def get_segmentation_mask(self, segmentation):
        '''Преобразует сегментационные данные в формат RLE и возвращает их.'''
        segmentation = np.clip(segmentation, 0, 1)
        poly_mask = mask.encode(segmentation, 'rle')

        return poly_mask

    def get_rle_dict(self, poly_mask):
        '''Возвращает словарь rle_dict, содержащий информацию о сегментации в виде количества (counts) и размера (size).'''
        rle_dict = {"counts": poly_mask.counts, "size": poly_mask.size}
        return rle_dict

    def get_json_dict(self):
        '''Инициализирует словарь json_dict, который будет использоваться для создания JSON-структуры данных для некоторого вида задачи или хранения информации. '''
        json_dict = {"info": {"date_created": str(datetime.datetime.now())}, "categories": []}
        face = {"id": 1, "name": "face"}
        left_eye = {"id": 2, "name": "left_eye"}
        right_eye = {"id": 3, "name": "right_eye"}
        json_dict["categories"].extend([face, left_eye, right_eye])
        json_dict["images"] = []
        json_dict["annotations"] = []
        return json_dict

    def get_image_dict(self, render_counter):
        '''Создает и возвращает словарь image_dict, который представляет информацию об изображении.'''
        image_dict = {"id": render_counter, "file_name": f"{render_counter}.png", "height": self.ypix, "width": self.xpix, "resolution": [self.xpix, self.ypix]}
        return image_dict

    def get_annotation_dict(self, render_counter, category_id, bbox, segmentation, angles):
        '''
        Создает и возвращает словарь аннотации, содержащий информацию об объекте на изображении. 

        - render_counter: идентификатор изображения
        - category_id: идентификатор категории объекта
        - bbox: ограничивающий прямоугольник объекта на изображении
        - segmentation: информация о сегментации объекта
        - angles: углы объекта на изображении
        '''
        annotation_dict = {"id": uuid.uuid4().int,
                           "image_id": render_counter, "category_id": category_id, "bbox": bbox,
                           "segmentation": segmentation, "area": 1, "angles": angles, "iscrowd": 0}
        return annotation_dict

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
    

    def rotmat_to_angles(self, m):
        """"Функция преобразует матрицу поворота в углы Эйлера (углы поворота вокруг осей X, Y и Z). 
        Она использует входную матрицу m и выполняет проверки, чтобы определить углы поворота. 
        Затем она возвращает углы в градусах в виде массива."""
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

    def draw_angles(self, image, pos, angles, line_size=40, thickness=2):
        "angles: (pitch, yaw, roll)"

        pos = np.array(pos, dtype=int)
        axis = (line_size*self.get_axes(angles)).astype(int)

        for (point, color) in zip(axis, [self.RED, self.GREEN, self.BLUE]):
            cv2.line(image, tuple(pos), tuple(pos + point), color, thickness)

        return image
    
    def rotate_points(self, points, angles):
        """
        Вращение 3D-точек под углами.

        Последовательность вращения: OZ (roll), OY (yaw), OX (pitch).

        Аргументы:
            - точки (np.array[:, 3]):
                ключевые точки [[x1, y1, z1], ...]
            - углы (np.array[3]):
                углы [pitch, yaw, roll], в градусах

        Возврат:
            повернутые точки, np.array[:, 3]
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
    
    def get_axes(self, angles):
        """
        Получить проецируемые оси

        Аргументы:
            - углы (np.array[3]):
                углы [pitch, yaw, roll], в градусах

        Возврат:
            Точки осей (np.array[3, 2]): OX, OY, OZ
        """
        return self.rotate_points(np.array([
            [1, 0, 0], [0, 1, 0], [0, 0, -1]
        ]), np.array(angles))[:, :2]
    
    def rotate_angles(self, angles, roll=0, yaw=0, pitch=0):
        """
        Поворот углов на угол.

        Аргументы:
            - углы (np.array[3]):
                углы [pitch, yaw, roll], в градусах
            - pitch, yaw, roll (float):
                углы, в градусах

        Возврат:
            углы поворота, np.array[3], в градусах
        """
        rotate = self.rotate_points(self.rotate_points(np.eye(3), angles), [pitch, yaw, roll])
        return self.rotmat_to_angles(rotate).astype(np.float32)
    

    def main_rendering_loop(self, rot_step):
        '''
        Эта функция представляет собой основной алгоритм, описанный в руководстве. Она принимает
        шаг вращения в качестве входных данных и выводит изображения и метки в указанные выше места.
        '''
        # Рассчитайте количество изображений и меток для создания
        n_renders = self.calculate_n_renders(rot_step)  # Рассчитать количество изображений
        print('Number of renders to create:', n_renders)

        
        
        counter = 1 #счётчик количества рендеров
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

                    for gamma in range(self.gamma_limits[0], self.gamma_limits[1] + 1, rotation_step):  # Петля для изменения угла гаммы
                        # вращение глаз и лица
                        for angle_x in range(-7, 10, 7):
                            self.eye_l.rotation_euler.x = m.radians(angle_x)
                            self.eye_r.rotation_euler.x = m.radians(angle_x)
                            self.eye_l_1.rotation_euler.x = m.radians(angle_x)
                            self.eye_r_1.rotation_euler.x = m.radians(angle_x)

                            for x in range(-20, 20, 15):
                                self.face.rotation_euler.x = m.radians(x)
                                
                                for y in range(-10, 10, 7):
                                    self.face.rotation_euler.y = m.radians(y)
                                    
                                    for angle_z in range(345, 375, 20):
                                        self.eye_l.rotation_euler.z = m.radians(angle_z)
                                        self.eye_r.rotation_euler.z = m.radians(angle_z)
                                        self.eye_l_1.rotation_euler.z = m.radians(angle_z)
                                        self.eye_r_1.rotation_euler.z = m.radians(angle_z)
                                        
                                    
                                        for z in range(-20, 20, 9):
                                            self.face.rotation_euler.z = m.radians(z)
                                            
                                            if counter > 3:
                                                exit()
            
                                            render_counter += 1  # Обновить счетчик
                                            
                                            ## Обновить вращение оси
                                            axis_rotation = (m.radians(beta_r), 0, m.radians(gamma))
                                            # Отображение демонстрационной информации — света
                                            print("On render:", render_counter)
                                            print("--> Location of the ligth:")
                                            print("     d:", d / 10, "m")
                                            print("     Beta:", str(beta_r) + "Degrees")
                                            print("     Gamma:", str(gamma) + "Degrees")

                                            ## Configure lighting можно поставить не рандом а например цикл
                                            energy = random.randint(0, 30)  # Захват случайной интенсивности света
                                            self.light.data.energy = energy  # Обновите <bpy.data.objects['Light']> energy information
                                            ## Создать рендер
                                            image_path = self.render_blender(render_counter)
                                            image = cv2.imread(image_path)  
 
                                            pos = np.array([500,500], dtype=int)
                                            angles_face = self.face.rotation_euler
                                            angles_face = [m.degrees(r) for r in angles_face]

                                            angles_face = [angles_face[0], -angles_face[2], angles_face[1]]
                                            image = self.draw_angles(image, pos, angles_face)
                                            cv2.imwrite(f"D:\blender\renders\{render_counter}.png", image)

                                            pos_l = np.array([300,200], dtype=int)
                                            angles_l = self.eye_l_1.rotation_euler
                                            angles_l = [m.degrees(r) for r in angles_l]

                                            angles_l = [angles_l[0], -angles_l[2], angles_l[1]]
                                            image = self.draw_angles(image, pos_l, angles_l)
                                            cv2.imwrite(f"D:\blender\renders\{render_counter}.png", image)

                                            pos_r = np.array([400,200], dtype=int)
                                            angles_r = self.eye_r_1.rotation_euler
                                            angles_r = [m.degrees(r) for r in angles_r]

                                            angles_r = [angles_r[0], -angles_r[2], angles_r[1]]
                                            image = self.draw_angles(image, pos_r, angles_r)
                                            cv2.imwrite(f"D:\blender\renders\{render_counter}.png", image)

                                            # Сфотографируйте текущую сцену и выведите файл render counter.png
                                            # Отображение демонстрационной информации — Информация о фотографии
                                            print("--> Picture information:")
                                            print("     Resolution:", (self.xpix * self.percentage, self.ypix * self.percentage))
                                            print("     Rendering samples:", self.samples)

                                            ## Лейблы
                                            if render_counter == 1:
                                                text_file_name = self.labels_filepath + '\\' + "jsonfile" + '.json'  # Создать имя файла этикетки
                                                text_file = open(text_file_name, 'w+')  # Открыть .txt-файл этикетки
                                                json_dict = self.get_json_dict()
                                                
                                                # Получить отформатированные координаты ограничивающих рамок всех объектов сцены
                                                # Отображение демонстрационной информации - Создание этикетки
                                            print("---> Label Construction")
                                            

                                            ### МАСКИ ###
                                            face_segmentation = self.get_obj_mask(["skin"])
                                            
                                            cv2.imwrite(f"D:\blender\masks\face\{render_counter}.png",
                                                        face_segmentation)
                                            left_eye_segmentation = self.get_obj_mask(["eye_l_1"])
                                            
                                            cv2.imwrite(f"D:\blender\masks/left_eye\{render_counter}.png",
                                                        left_eye_segmentation)
                                            right_eye_segmentation = self.get_obj_mask(["eye_r_1"])
                                            
                                            cv2.imwrite(f"D:\blender\masks/right_eye\{render_counter}.png",
                                                        right_eye_segmentation)
                                            
                                            face_mask = self.get_segmentation_mask(face_segmentation)
                                            left_eye_mask = self.get_segmentation_mask(left_eye_segmentation)
                                            right_eye_mask = self.get_segmentation_mask(right_eye_segmentation)

                                            face_rle_dict = self.get_rle_dict(face_mask)
                                            left_eye_rle_dict = self.get_rle_dict(left_eye_mask)
                                            right_eye_rle_dict = self.get_rle_dict(right_eye_mask)

                                            ### ББОКСЫ ###

                                            face_bbox = self.get_bbox_from_rle(face_mask)
                                            left_eye_bbox = self.get_bbox_from_rle(left_eye_mask)
                                            right_eye_bbox = self.get_bbox_from_rle(right_eye_mask)

                                            bboxes_dict = {"face": self.get_list_from_bbox_obj(face_bbox),
                                                            "left_eye": self.get_list_from_bbox_obj(left_eye_bbox),
                                                            "right_eye": self.get_list_from_bbox_obj(right_eye_bbox)}

                                            image = cv2.imread(f"D:\blender\renders\{render_counter}.png")
                                            for key in bboxes_dict.keys():
                                                bbox = bboxes_dict[key]
                                                i = 255
                                                x, y, width, height = map(int, bbox)
                                                if "eye" in key:
                                                    i = 0
                                                cv2.rectangle(image, (x, y), (x + width, y + height),
                                                                (255, 255, i), 2)

                                            cv2.imwrite(f"D:\blender\bbox\{render_counter}.png",
                                                        image)

                                            ### JSON ФАЙЛЫ ###
                                            curr_image_dict = self.get_image_dict(render_counter)

                                            face_anno_dict = self.get_annotation_dict(render_counter, 1,
                                                                                        bboxes_dict["face"],
                                                                                        face_rle_dict, angles_face)
                                            left_eye_anno_dict = self.get_annotation_dict(render_counter, 2,
                                                                                            bboxes_dict["left_eye"],
                                                                                            left_eye_rle_dict, angles_l)
                                            right_eye_anno_dict = self.get_annotation_dict(render_counter, 3,
                                                                                            bboxes_dict["right_eye"],
                                                                                            right_eye_rle_dict, angles_r)

                                            json_dict["images"].append(curr_image_dict)
                                            json_dict["annotations"].extend([face_anno_dict, left_eye_anno_dict, right_eye_anno_dict])

                                            if render_counter == 20: #это значение равно значению counter
                                                json.dump(json_dict, text_file)
                                                text_file.close()  # Закройте файл .txt, соответствующий метке.
                                            
                                            print('Progress =', str(render_counter) + '/' + str(n_renders))
                                            report.write('Progress: ' + str(render_counter) + ' Rotation: ' + str(axis_rotation) + ' z_d: ' + str(d / 10) + '\n')
                                            counter += 1
                                
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
        return f"D:\blender\renders\{image_name}"
        
    def export_render(self, res_x, res_y, res_per, samples, file_path, file_name):
        # Установите все параметры сцены
        bpy.context.scene.cycles.samples = samples
        self.scene.render.resolution_x = res_x
        self.scene.render.resolution_y = res_y
        self.scene.render.resolution_percentage = res_per
        self.scene.render.filepath = file_path + '/' + file_name

        #Сфотографировать текущую видимую сцену
        bpy.ops.render.render(write_still=True)

    def calculate_n_renders(self, rotation_step): #Функция подсчитывает количество рендеров
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

## Запустить генерацию данных
if __name__ == '__main__':
    # Инициализировать класс рендеринга как r
    r = Render()
    # Инициализировать свет
    r.set_light()
    # Начать генерацию данных
    rotation_step = 100
    r.main_rendering_loop(rotation_step)
