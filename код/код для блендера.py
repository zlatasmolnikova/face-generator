import bpy
import numpy as np
import math as m
import random

## Main Class
class Render:
    def __init__(self):
        ## Scene information
        # Define the scene information
        self.scene = bpy.data.scenes['Scene']
        # Define the information relevant to the <bpy.data.objects>
        self.set_camera = bpy.data.objects['Camera']
        self.light = bpy.data.objects['Light']
        self.obj_names = ['mmm_FaceMesh_LodGroup']
        self.objects = self.create_objects() # Create list of bpy.data.objects from bpy.data.objects[1] to bpy.data.objects[N]

        ## Render information
        self.light_d_limits = [0.2, 0.8] # Define range of heights z in m that the camera is going to pan through
        self.beta_limits = [80, -80] # Define range of beta angles that the camera is going to pan through
        self.gamma_limits = [0, 360] # Define range of gamma angles that the camera is going to pan through
        
        ## Output information
        # Input your own preferred location for the images and labels
        self.images_filepath = 'D:\чел'
        self.labels_filepath = 'D:\чел'

    def set_light(self):
        self.light.location = (0, 0, 3)

    def main_rendering_loop(self, rot_step):
        '''
        This function represent the main algorithm explained in the Tutorial, it accepts the
        rotation step as input, and outputs the images and the labels to the above specified locations.
        '''
        ## Calculate the number of images and labels to generate
        n_renders = self.calculate_n_renders(rot_step) # Calculate number of images
        print('Number of renders to create:', n_renders)

        #accept_render = input('\nContinue?[Y/N]:  ') # Ask whether to procede with the data generation
        accept_render = 'Y'
        if accept_render == 'Y': # If the user inputs 'Y' then procede with the data generation
            # Create .txt file that record the progress of the data generation
            report_file_path = self.labels_filepath + 'progress_report.txt'
            report = open(report_file_path, 'w')
            # Multiply the limits by 10 to adapt to the for loop
            dmin = int(self.light_d_limits[0] * 10)
            dmax = int(self.light_d_limits[1] * 10)
            # Define a counter to name each .png and .txt files that are outputted
            render_counter = 0
            # Define the step with which the pictures are going to be taken
            rotation_step = rot_step

            # Begin nested loops
            for d in range(dmin, dmax + 1, 2): # Loop to vary the height of the camera
                ## Update the height of the camera
                self.light.location = (0, 0, d/10) # Divide the distance z by 10 to re-factor current height

                # Refactor the beta limits for them to be in a range from 0 to 360 to adapt the limits to the for loop
                min_beta = (-1)*self.beta_limits[0] + 90
                max_beta = (-1)*self.beta_limits[1] + 90

                for beta in range(min_beta, max_beta + 1, rotation_step): # Loop to vary the angle beta
                    beta_r = (-1)*beta + 90 # Re-factor the current beta

                    for gamma in range(self.gamma_limits[0], self.gamma_limits[1] + 1, rotation_step): # Loop to vary the angle gamma
                        render_counter += 1 # Update counter
                        
                        ## Update the rotation of the axis
                        axis_rotation = (m.radians(beta_r), 0, m.radians(gamma)) 
                        #self.axis.rotation_euler = axis_rotation # Assign rotation to <bpy.data.objects['Empty']> object
                        # Display demo information - Location of the camera
                        print("On render:", render_counter)
                        print("--> Location of the ligth:")
                        print("     d:", d/10, "m")
                        print("     Beta:", str(beta_r)+"Degrees")
                        print("     Gamma:", str(gamma)+"Degrees")

                        ## Configure lighting можно поставить не рандом а например цикл
                        energy = random.randint(0, 30) # Grab random light intensity
                        self.light.data.energy = energy # Update the <bpy.data.objects['Light']> energy information
       

                        ## Generate render
                        self.render_blender(render_counter) # Take photo of current scene and ouput the render_counter.png file
                        # Display demo information - Photo information
                        print("--> Picture information:")
                        print("     Resolution:", (self.xpix*self.percentage, self.ypix*self.percentage))
                        print("     Rendering samples:", self.samples)

                        ## Output Labels
                        text_file_name = self.labels_filepath + '/' + str(render_counter) + '.json' # Create label file name
                        text_file = open(text_file_name, 'w+') # Open .txt file of the label
                        # Get formatted coordinates of the bounding boxes of all the objects in the scene
                        # Display demo information - Label construction
                        print("---> Label Construction")
                        #text_coordinates = self.get_all_coordinates()
                        #splitted_coordinates = text_coordinates.split('\n')[:-1] # Delete last '\n' in coordinates
                        #text_file.write('\n'.join(splitted_coordinates)) # Write the coordinates to the text file and output the render_counter.txt file
                        #text_file.close() # Close the .txt file corresponding to the label
                        ## Show progress on batch of renders
                        print('Progress =', str(render_counter) + '/' + str(n_renders))
                        report.write('Progress: ' + str(render_counter) + ' Rotation: ' + str(axis_rotation) + ' z_d: ' + str(d / 10) + '\n')

            report.close() # Close the .txt file corresponding to the report

    def render_blender(self, count_f_name):
        # Define random parameters
        random.seed(random.randint(1,10))
        self.xpix = 1000
        self.ypix = 1000
        self.percentage = 100
        self.samples = 15
        # Render images
        image_name = str(count_f_name) + '.png'
        self.export_render(self.xpix, self.ypix, self.percentage, self.samples, self.images_filepath, image_name)

    def export_render(self, res_x, res_y, res_per, samples, file_path, file_name):
        # Set all scene parameters
        bpy.context.scene.cycles.samples = samples
        self.scene.render.resolution_x = res_x
        self.scene.render.resolution_y = res_y
        self.scene.render.resolution_percentage = res_per
        self.scene.render.filepath =  file_path + '/' + file_name

        # Take picture of current visible scene
        bpy.ops.render.render(write_still=True)

    def calculate_n_renders(self, rotation_step):
        zmin = int(self.light_d_limits[0] * 10)
        zmax = int(self.light_d_limits[1] * 10)

        render_counter = 0
        rotation_step = rotation_step

        for d in range(zmin, zmax+1, 2):
            light_location = (0,0,d/10)
            min_beta = (-1)*self.beta_limits[0] + 90
            max_beta = (-1)*self.beta_limits[1] + 90

            for beta in range(min_beta, max_beta+1,rotation_step):
                beta_r = 90 - beta

                for gamma in range(self.gamma_limits[0], self.gamma_limits[1]+1,rotation_step):
                    render_counter += 1
                    axis_rotation = (beta_r, 0, gamma)

        return render_counter

    def create_objects(self):  # This function creates a list of all the <bpy.data.objects>
        objs = []
        for obj in self.obj_names:
            objs.append(bpy.data.objects[obj])

        return objs

## Run data generation
if __name__ == '__main__':
    # Initialize rendering class as r
    r = Render()
    # Initialize light
    r.set_light()
    # Begin data generation
    rotation_step = 120 #(360/x)*(160/x)*4
    r.main_rendering_loop(rotation_step)
