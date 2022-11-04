import PySimpleGUI as sg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from VectorPlot import create_plot, create_empty_plot
from VectorUtils import *


AppFont = 'Any 16'
sg.theme('LightGrey')

exit_layout = [
    [sg.Button('Exit', font=AppFont)]
]

vector_input_layout = [
    [sg.Text('V', key='V', size=(2, 1)),
     sg.InputText(key='V_BOX', size=(5, 1)),
     sg.Radio('2D', 'VectorDimension', key='2D', enable_events=True, default=True),
     sg.Checkbox('Show Cross Product', key='CP', visible=False, default=False)],
    [sg.Text('U', key='U', size=(2, 1)),
     sg.InputText(key='U_BOX', size=(5, 1)),
     sg.Radio('3D', 'VectorDimension', key='3D', enable_events=True, default=False)],
    [sg.Button('Update', font=AppFont)]
]

overall_layout = [
    [sg.Canvas(key='figCanvas')],
    [
        sg.Column(vector_input_layout, element_justification='left', expand_x=True),
        sg.VSeparator(),
        sg.Column(exit_layout, element_justification='right', expand_x=True)
     ]
]


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def update_figure(window, V):
    vec_list = [[float(x) for x in i.strip().split(',')] for i in V]
    if window['CP']:
        vec_list.append(Vector.cross_product(vec_list[0], vec_list[1]))

    return draw_figure(window['figCanvas'].TKCanvas, create_plot(vec_list))


def main():
    window = sg.Window('Vector Plotting',
                       overall_layout,
                       finalize=True,
                       resizable=True)
    figure = draw_figure(window['figCanvas'].TKCanvas, create_plot([[1,2], [2,1], [1,1]]))
    while True:
        event, values = window.read()
        # print('event:')
        # print(event)
        # print('value:')
        # print(values)
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        if event == 'Update':
            figure.get_tk_widget().forget()
            plt.close('all')
            update_figure(window, [values['V_BOX'], values['U_BOX']])
        elif event == '2D':
            window['CP'].Update(visible=False)
            window['CP'].Update(False)
        elif event == '3D':
            window['CP'].Update(visible=True)

    window.close()


if __name__ == '__main__':
    main()