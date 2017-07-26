import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from IPython.display import display, Math, Latex, HTML
from ipywidgets import interact, IntSlider, interactive
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Title
from bokeh.io import push_notebook, show, output_notebook
from bokeh.layouts import gridplot, row



class Visualization(object):
    def __init__(self, total_epoch, batch_size, printing):
        #This is the visualazation library used for this notebook
        self.total_epoch=total_epoch
        self.batch_size=batch_size
        self.printing=printing
    
    def plotting_cost(self, cost_list):
        '''
        Function plots cost
        
        Paramaters:
        __________
        
        cost_list: list of error cost/loss during the training
        
    
        '''
        
        fig, axes = plt.subplots(figsize=(4, 5), dpi=100)
        axes.plot(np.arange(len(cost_list))*self.printing, cost_list, color='red', lw=2, alpha=0.4)
        axes.scatter(np.arange(len(cost_list))*self.printing, cost_list, color='black', alpha=0.9)
        axes.set_xlabel('Epoch')
        axes.set_ylabel('ELBO')
        axes.grid(alpha=0.09, color='grey')
        plt.tight_layout()
        
    def plotting_latent_space(self, z_space, y_labels):
        
        '''
        Function plots changes in Z space during the training 
        
        Paramaters:
        __________
        
        z_space: list contaning numpy array over training, Dimensions [training epochs x  2D].
                 Z_space must be 2-dimensiol, if Z-space is 3D or hight, transfrom it to 2D via PCA or 
                 tSNE
        
        y_labels: labels for data that was passed via Z space, labels must be binary matrix 
                  [number_samples x  number of total class] 
        
    
        '''
        #color_dictionary
        self.z_space=z_space
        self.y_labels=y_labels
        self.c_dict={0: 'blue', 
                1: 'pink', 
                2: 'green',
                4: "red",
                5: "olive", 
                6: "darkred", 
                7: "grey", 
                8: "skyblue", 
                9: "orange"}

        #listt of colors per position
        self.k=list(map(self.c_dict.get, list(np.argmax(y_labels, axis=1))))
        
        #data_column for feeding and updating in to the graph
        source = ColumnDataSource(dict(
            x=self.z_space[0][:, 1],
            y=self.z_space[0][:, 0],
            color=self.k, 
            labels=np.argmax(self.y_labels, axis=1),
            line_color=self.k
        ))
        p = figure(plot_width=650, 
                   plot_height=650, 
                   x_range=(-6.5, 6.5), 
                   y_range=(-6.5, 6.5), 
                   logo=None)
        
        p.circle(x='x',
                 y='y', 
                 size=10, 
                 line_color='line_color', 
                 legend='labels', 
                 color='color',
                 alpha=0.4, 
                 source=source)
        
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.toolbar_location = None
        p.xaxis.minor_tick_line_color = None
        p.xgrid.grid_line_color = 'grey'
        p.xgrid.grid_line_alpha = 0.1
        p.xaxis.bounds = (-6.5, 6.5)
        p.yaxis.major_tick_line_width = 3
        p.yaxis.minor_tick_line_color = None
        p.ygrid.grid_line_color = 'grey'
        p.ygrid.grid_line_alpha = 0.1
        p.yaxis.bounds = (-6.5, 6.5)
        p.yaxis.major_tick_line_width = 3
        p.legend.location = "top_right"
        p.legend.orientation = "vertical"
        
        show(p, notebook_handle=True)
        
        def update(g):
            w=g
            new_data = dict(x=self.z_space[w][:, 0], 
                            y=self.z_space[w][:, 1],
                            color=self.k, 
                            labels=np.argmax(self.y_labels, axis=1), 
                            line_color=self.k)
            source.data = new_data
            p.title.text = str(' number of epoch: ' + str(w))
            push_notebook()


        slider = IntSlider(min=0, max=(self.total_epoch//self.printing)-1, step=1, description='Epoch')
        interact(update,  g=slider);
        
    def plotting_training_visual(self, reconstruction, true_x):
        '''
        Function accepts true x datasate, and uses NN netwrok to reconstruct its images over 
        the tranings, size of true_x and reconstruction matrices should be 
        [batch_size x vectrized image]
        
        Paramaters:
        __________
        
        reconstruction: result of true_data passed via neural netwrok
        
        true_x: data that was passed for generating reconstruction, must be flatten image
        
        '''
        
        x_reshaped=np.array(reconstruction).reshape(self.total_epoch//self.printing, self.batch_size, 28, 28)
        x_reshaped = x_reshaped[:,:,::-1]
        
        x_true=true_x[:self.batch_size].reshape(self.batch_size, 28, 28)
        x_true= x_true[:,::-1]
        
        custum_blue=['#ffffff','#f0f0f0','#d9d9d9', '#a6bddb','#74a9cf','#3690c0','#0570b0','#045a8d','#023858']
        custum_grey=['#ffffff','#f0f0f0','#d9d9d9','#bdbdbd','#969696','#737373','#525252','#252525', '#000000']

        r1, r2, r3, r4=np.random.randint(0, self.batch_size, 4)
        figure_settings={'toolbar_location': None, 
              'plot_width': 230,
              'plot_height': 230,
              'x_range': (0, 28), 
              'y_range': (0, 28),
              'logo': None}

        image_settings_target={'x': 0, 
                        'y': 0, 
                        'dw': 28,
                        'dh': 28,
                        'palette': custum_grey}

        image_settings_recons={'x': 0, 
                        'y': 0, 
                        'dw': 28,
                        'dh': 28,
                        'palette': custum_blue}

        p1 = figure(**figure_settings)
        p1.image(image=[x_reshaped[0][r1]], **image_settings_recons)
        p1.axis.visible=False
        p1.add_layout(Title(text="Reconstruction", align="center", text_font_size = "18px"), "left")

        p2 = figure(**figure_settings)
        p2.image(image=[x_reshaped[0][r2]], **image_settings_recons)
        p2.axis.visible=False

        p3 = figure(**figure_settings)
        p3.image(image=[x_reshaped[0][r3]], **image_settings_recons)
        p3.axis.visible=False
        
        p4 = figure(**figure_settings)
        p4.image(image=[x_reshaped[0][r4]], **image_settings_recons)
        p4.axis.visible=False

        p5 = figure(**figure_settings)
        p5.image(image=[x_true[r1]], **image_settings_target)
        p5.axis.visible=False
        p5.add_layout(Title(text="Target", align="center", text_font_size = "18px", ), "left")

        p6 = figure(**figure_settings)
        p6.image(image=[x_true[r2]], **image_settings_target)
        p6.axis.visible=False

        p7 = figure(**figure_settings)
        p7.image(image=[x_true[r3]], **image_settings_target)
        p7.axis.visible=False
        
        p8 = figure(**figure_settings)
        p8.image(image=[x_true[r4]], **image_settings_target)
        p8.axis.visible=False

        grid = gridplot([[p5, p6, p7, p8], [p1, p2, p3, p4]])
        show(grid, notebook_handle=True)
        output_notebook()
        
        def update(g):
            
            w=g
            i, i_1, i_2, i_3 =x_reshaped[w][r1], x_reshaped[w][r2], x_reshaped[w][r3], x_reshaped[w][r4]
            p1.image(image=[i], **image_settings_recons)
            p2.image(image=[i_1], **image_settings_recons)
            p3.image(image=[i_2], **image_settings_recons)
            p4.image(image=[i_3], **image_settings_recons)
            push_notebook()

        slider_1 = IntSlider(min=0, max=(self.total_epoch//self.printing)-1, step=1, description='Epochs')
        interact(update,  g=slider_1);
        push_notebook()
    
    
    
    def plott_gener_sample_from_dist(self, va, left_x=3, right_x=4, top_y=1, bottom_y=-2, img_size=28):
        
        '''
        Function accepts true z mean boundries, to generate new sample from trainied
        distribution
        
        Paramaters:
        __________
        
        left_x, right_x, top_y, bottom_y: dimensiopn of the box that are used for sampling

        '''
        
        cell_x = cell_y = 30
        canvas_width=img_size*cell_x
        canvas_hight=img_size*cell_y
        
        x_values, y_values = np.linspace(left_x, right_x, cell_x), np.linspace(bottom_y, top_y, cell_y)
        canvas = np.empty((canvas_width, 28*cell_y))
        for i, yi in enumerate(x_values):
            for j, xi in enumerate(y_values):
                x_mean = va.generating_samples(np.array([[xi, yi]]*self.batch_size))
                canvas[(cell_x-i-1)*img_size:(cell_y-i)*img_size, j*img_size:(j+1)*img_size] = x_mean[0].reshape(img_size, img_size)
        
        fig, axes = plt.subplots(figsize=(10, 10), dpi=100)     
        axes.imshow(canvas, origin="upper", cmap="Purples")
        axes.set_xticks([])
        axes.set_yticks([])
        axes.spines['bottom'].set_color('purple')
        axes.spines['top'].set_color('purple') 
        axes.spines['right'].set_color('purple')
        axes.spines['left'].set_color('purple')
        
        #box plot
        source = ColumnDataSource(dict(
            x=self.z_space[-1][:, 1],
            y=self.z_space[-1][:, 0],
            color=self.k, 
            labels=np.argmax(self.y_labels, axis=1),
            line_color=self.k
        ))
        
        p = figure(plot_width=500, 
                   plot_height=500, 
                   x_range=(-6.5, 6.5), 
                   y_range=(-6.5, 6.5), 
                   logo=None)
        
        p.circle(x='x',
                 y='y', 
                 size=10, 
                 line_color='line_color', 
                 legend='labels', 
                 color='color',
                 alpha=0.4, 
                 source=source)
        
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.toolbar_location = None
        p.xaxis.minor_tick_line_color = None
        p.xgrid.grid_line_color = 'grey'
        p.xgrid.grid_line_alpha = 0.1
        p.xaxis.bounds = (-6.5, 6.5)
        p.yaxis.major_tick_line_width = 3
        p.yaxis.minor_tick_line_color = None
        p.ygrid.grid_line_color = 'grey'
        p.ygrid.grid_line_alpha = 0.1
        p.yaxis.bounds = (-6.5, 6.5)
        p.yaxis.major_tick_line_width = 3
        p.legend.location = "top_right"
        p.legend.orientation = "vertical"
        
        p.quad(top=[y_values.min()], 
               bottom=[y_values.max()], 
               left=[x_values.min()],
               right=[x_values.max()], 
               color="black", 
               line_width=4, 
               line_color='purple', 
               alpha=0.1)
        
        show(p, notebook_handle=True)
