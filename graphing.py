'''
Several no-fuss methods for creating plots 
This code is from MicrosoftDocs, originally authored by Lee Reid
'''
from colorsys import yiq_to_rgb
from distutils.log import error
from typing import Optional, Callable, Union, List
from numpy import exp, repeat
import numpy
import pandas
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as graph_objects

# Set the default theme 
template =  graph_objects.layout.Template() 
template.layout = graph_objects.Layout(
                                    title_x=0.5,
                                    # border width and size
                                    margin=dict(l=2, r=2, b=2, t=30),
                                    height=400,
                                    # Interaction
                                    hovermode="closest",
                                    # axes
                                    xaxis_showline=True,
                                    xaxis_linewidth=2,
                                    yaxis_showline=True,
                                    yaxis_linewidth=2,
                                    # Pick a slightly different P.O.V from default
                                    # this avoids the extremities of the y and x axes
                                    # being cropped off
                                    scene_camera=dict(eye=dict(x=1.5, y=1.5, z=0.1))
                                    )    
                                    
template.data.scatter = [graph_objects.Scatter(marker=dict(opacity=0.8))]
template.data.scatter3d = [graph_objects.Scatter3d(marker=dict(opacity=0.8))]
template.data.surface = [graph_objects.Surface()]
template.data.histogram = [graph_objects.Histogram(marker=dict(line=dict(width=1)))]
template.data.box = [graph_objects.Box(boxpoints='outliers', notched=False)]


pio.templates["custom_template"] = template
pio.templates.default = "plotly_white+custom_template"


def _to_human_readable(text:str):
    '''
    Converts a label into a human readable form
    '''
    return text.replace("_", " ")


def _prepare_labels(df:pandas.DataFrame, labels:List[Optional[str]], replace_nones:bool=True):
    '''
    Ensures labels are human readable. 
    Automatically picks data if labels not provided explicitly
    '''

    human_readable = {}

    if isinstance(replace_nones, bool):
        replace_nones = [replace_nones] * len(labels) 

    for i in range(len(labels)):
        lab = labels[i]
        if replace_nones[i] and (lab is None):
            lab = df.columns[i]
            labels[i] = lab

        # make human-readable
        if lab is not None:
            human_readable[lab] = _to_human_readable(lab)
    
    return labels, human_readable


def box_and_whisker(df:pandas.DataFrame, 
                label_x:Optional[str]=None, 
                label_y:Optional[str]=None, 
                label_x2:Optional[str]=None,
                title=None, 
                show:bool=False):
    '''
    Creates a box and whisker plot and optionally shows it. Returns the figure for that plot.
    Note that if calling this from jupyter notebooks and not capturing the output
    it will appear on screen as though `.show()` has been called
    df: The data
    label_x: What to group by. Defaults to None
    label_y: What to plot on the y axis. Defaults to count of df.columns[0]
    label_x2: If provided, splits boxplots into 2+ per x value, each with its own colour
    title: Plot title
    show:   appears on screen. NB that this is not needed if this is called from a
            notebook and the output is not captured 
    '''

    # Automatically pick columns if not specified
    selected_columns, axis_labels = _prepare_labels(df, [label_x, label_y, label_x2], replace_nones=[False, True, False])

    fig = px.box(df,
                    x=selected_columns[0],
                    y=selected_columns[1],
                    color=label_x2,
                    labels=axis_labels,
                    title=title)

    # Show the plot, if requested
    if show:
        fig.show()
    
    # return the figure
    return fig


def histogram(df:pandas.DataFrame, 
                label_x:Optional[str]=None, 
                label_y:Optional[str]=None, 
                label_colour:Optional[str]=None,
                nbins:Optional[int]=None,
                title=None, 
                include_boxplot=False,
                histfunc:Optional[str]=None,
                show:bool=False):
    '''
    Creates a 2D histogram and optionally shows it. Returns the figure for that histogram.
    Note that if calling this from jupyter notebooks and not capturing the output
    it will appear on screen as though `.show()` has been called
    df: The data
    label_x: What to bin by. Defaults to df.columns[0]
    label_y: If provided, the sum of these numbers becomes the y axis. Defaults to count of label_x
    label_colour: If provided, creates a stacked histogram, splitting each bar by this column
    title: Plot title
    nbins: the number of bins to show. None for automatic
    histfunc: How to calculate y. See plotly for options
    show:   appears on screen. NB that this is not needed if this is called from a
            notebook and the output is not captured 
    '''

    if type(df) is numpy.array or type(df) == pandas.core.series.Series:
        # Convert into a suitable format
        if(type(df) == pandas.core.series.Series or df.ndim == 1):
            d = dict()
            if label_x is None:
                label_x = "value"
            d[label_x] = df
            df = d
        else:
            raise Exception("Data input should be a pandas dataframe, or a 1D numpy array if no labels are provided")

    # Automatically pick columns if not specified
    selected_columns, axis_labels = _prepare_labels(df, [label_x, label_y, label_colour], replace_nones=[True, False, False])


    fig = px.histogram(df,
                        x=selected_columns[0],
                        y=selected_columns[1],
                        nbins=nbins,
                        color=label_colour,
                        labels=axis_labels,
                        title=title,
                        marginal="box" if include_boxplot else None,
                        histfunc=histfunc
                        )

    # Show the plot, if requested
    if show:
        fig.show()
    
    # return the figure
    return fig


def multiple_histogram(df:pandas.DataFrame, 
                label_x:str, 
                label_group:str,
                label_y:Optional[str]=None,
                histfunc:str='count',
                nbins:Optional[int]=None,
                title=None,
                show:bool=False):
    '''
    Creates a 2D histogram and optionally shows it. Returns the figure for that histogram.
    Note that if calling this from jupyter notebooks and not capturing the output
    it will appear on screen as though `.show()` has been called
    df: The data
    label_x: What to bin by. Defaults to df.columns[0]
    label_y: If provided, the sum of these numbers becomes the y axis. Defaults to count of label_x
    title: Plot title
    nbins: the number of bins to show. None for automatic
    show:   appears on screen. NB that this is not needed if this is called from a
            notebook and the output is not captured 
    '''

    assert (histfunc != 'count') or (label_y == None), "Set histfunc to a value such as sum or avg if using label_y"

    # Automatically pick columns if not specified
    selected_columns, axis_labels = _prepare_labels(df,  [label_x, label_y, label_group], replace_nones=[True, False, False])

    fig = graph_objects.Figure(layout=dict(
                                    title=title, 
                                    xaxis_title_text=axis_labels[label_x],
                                    yaxis_title_text=histfunc if label_y is None else (histfunc + " of " + axis_labels[label_y]))
                                )
    
    group_values = sorted(set(df[label_group]))

    for group_value in group_values:
        dat = df[df[label_group] == group_value]
        x = dat[selected_columns[0]]

        if label_y is None:
            y = None
        else:
            y = dat[selected_columns[1]]
            
        fig.add_trace(graph_objects.Histogram(
            x=x,
            y=y,
            histfunc=histfunc,
            name=group_value, # name used in legend and hover labels
            nbinsx=nbins))

    # fig = px.histogram(df,
    #                     x=selected_columns[2],
    #                     y=selected_columns[0],
    #                     nbins=nbins,
    #                     color=selected_columns[1],
    #                     labels=axis_labels,
    #                     # title=title,
    #                     # marginal="box" if include_boxplot else None
    #                     )

    # Show the plot, if requested
    if show:
        fig.show()
    
    # return the figure
    return fig


def equation_2d(trendline:Callable,
                x_min:Optional[float]=0,
                x_max:Optional[float]=1,
                label_x:Optional[str]='x',
                label_y:Optional[str]='y',
                show:Optional[bool]=False):
    '''
    Plots an equation on 2D axes
    '''

    evaluate_for = numpy.linspace(x_min, x_max, num=200)
    

    try:
        y_vals = trendline(evaluate_for)
    except:
        # probably stats models with formulae
        asDict = dict()
        asDict[label_x] = evaluate_for
        y_vals = trendline(asDict)

    data = pandas.DataFrame()
    data[label_x]=evaluate_for
    data[label_y]=_sanitise_trendline(y_vals)

    # Create the figure and plot
    fig = px.line(data, 
                x=label_x, 
                y=label_y, 
                )
    
    # Show the plot, if requested
    if show:
        fig.show()
    
    # return the figure
    return fig
    

def equation_2d_animated(trendline:Callable,
                animation_vars:List,
                x_min:Optional[float]=0,
                x_max:Optional[float]=1,
                label_x:Optional[str]='x',
                label_y:Optional[str]='y',
                label_animation:Optional[str]="z",
                show_play_button=False, # often has poor behaviour
                show:Optional[bool]=False):
    '''
    Plots an equation on 2D axes
    '''

    evaluate_for = numpy.linspace(x_min, x_max, num=200)
    
    all_x = numpy.array([])
    all_y = numpy.array([])

    for anim_var in animation_vars:
        y_vals = trendline(evaluate_for, anim_var)
        all_x = numpy.append(all_x, evaluate_for)
        all_y = numpy.append(all_y, _sanitise_trendline(y_vals))
        

    data = pandas.DataFrame()
    data[label_x]=all_x
    data[label_y]=all_y
    data[label_animation]=numpy.repeat(animation_vars, len(evaluate_for))

    # Create the figure and plot
    fig = px.line(data, 
                x=label_x, 
                y=label_y, 
                animation_frame=label_animation,
                range_y=[min(all_y), max(all_y)]
                # color=selected_columns[2],
                # labels=axis_labels,
                # hover_data=[label_size],
                # title=title,
                #line=dict(color="blue")
                )
    if not show_play_button:
        fig["layout"].pop("updatemenus")
    
    # Show the plot, if requested
    if show:
        fig.show()
    
    # return the figure
    return fig



def scatter_2D(df:pandas.DataFrame, 
                label_x:Optional[str]=None, 
                label_y:Optional[str]=None, 
                label_colour:Optional[str]=None,
                label_size:Optional[str]=None,
                size_multiplier:float=1,
                title=None, 
                show:bool=False,
                trendline:Union[Callable,List[Callable],None]=None,
                x_min:Optional[float]=None,
                x_max:Optional[float]=None):
    '''
    Creates a 2D scatter plot and optionally shows it. Returns the figure for that scatter.
    Note that if calling this from jupyter notebooks and not capturing the output
    it will appear on screen as though `.show()` has been called
    df: The data
    label_x: The label to extract from df to plot on the x axis. Defaults to df.columns[0]
    label_y: The label to extract from df to plot on the y axis. Defaults to df.columns[1]
    label_colour: The label to extract from df to colour points by
    title: Plot title
    show:   appears on screen. NB that this is not needed if this is called from a
            notebook and the output is not captured 
    trendline:  A function that accepts X (a numpy array) and returns Y (an iterable). If
            an ndimensional numpy array, it will be flattened automatically
    '''

    # Automatically pick columns if not specified
    selected_columns, axis_labels = _prepare_labels(df, [label_x, label_y, label_colour], [True, True, False])

    if x_min is None:
        x_min = min(df[selected_columns[0]])
    if x_max is None:
        x_max = max(df[selected_columns[0]])

    # Create the figure and plot
    fig = px.scatter(df, 
                x=selected_columns[0], 
                y=selected_columns[1], 
                color=selected_columns[2],
                labels=axis_labels,
                hover_data=[label_size],
                title=title,
                range_x=[x_min,x_max]
                )

    if label_size is None:
        # User a marker size inversely proportional to the number of points
        size = int(round((22.0 - 19/(1+exp(-(df.shape[0]/100-2)))) * size_multiplier))
    else:
        # Set the size based on a label
        size = df[label_size]*size_multiplier

    fig.update_traces(marker={'size': size})

    # Create trendlines
    if trendline is not None:
        if isinstance(trendline, Callable):
            trendline = [trendline]
        evaluate_for = numpy.linspace(x_min, x_max, num=200)
        shapes = []
        for t in trendline:

            try:
                y_vals = t(evaluate_for)
            except:
                # probably stats models with formulae
                asDict = dict()
                asDict[selected_columns[0]] = evaluate_for
                y_vals = t(asDict)

            y_vals = _sanitise_trendline(y_vals)

            path = "M" + " L ".join([str(c[0]) + " " + str(c[1]) for c in zip(evaluate_for,y_vals)])
            
            shapes.append(dict(
                                type="path",
                                path=path,
                                line_color="Crimson",
                            )
                        )
        
        fig.update_layout(shapes=shapes)

    # Show the plot, if requested
    if show:
        fig.show()
    
    # return the figure
    return fig


def _sanitise_trendline(val):
    if type(val) is numpy.ndarray and val.ndim != 1:
        # Flatten it. Some packages return arrays like [[1],[2],[3]]
        return val.flatten()
    return val


def scatter_3D(df:pandas.DataFrame, 
                label_x:Optional[str]=None, 
                label_y:Optional[str]=None, 
                label_z:Optional[str]=None, 
                label_colour:Optional[str]=None,
                title=None, 
                show:bool=False):
    '''
    Creates a 3D scatter plot and optionally shows it. Returns the figure for that scatter.
    Note that if calling this from jupyter notebooks and not capturing the output
    it will appear on screen as though `.show()` has been called
    df: The data
    label_x: The label to extract from df to plot on the x axis. Defaults to df.columns[0]
    label_y: The label to extract from df to plot on the y axis. Defaults to df.columns[1]
    label_z: The label to extract from df to plot on the z axis. Defaults to df.columns[2]
    label_colour: The label to extract from df to colour points by. Defaults to label_x
    title: Plot title
    show:   appears on screen. NB that this is not needed if this is called from a
            notebook and the output is not captured 
    '''

    # Automatically pick columns if not specified
    selected_columns, axis_labels = _prepare_labels(df, [label_x, label_y, label_z])

    if label_colour is None:
        # Colour by the Z dimension
        label_colour = selected_columns[2]
    else:
        axis_labels[label_colour] = _to_human_readable(label_colour)

    # Create the figure and plot
    fig = px.scatter_3d(df, 
                x=selected_columns[0], 
                y=selected_columns[1], 
                z=selected_columns[2],
                color=label_colour,
                labels=axis_labels,
                title=title)


    # Show the plot, if requested
    if show:
        fig.show()
    
    # return the figure
    return fig

def surface(x_values, 
            y_values, 
            calc_z:Callable,
            title=None, 
            axis_title_x:Optional[str]=None,
            axis_title_y:Optional[str]=None,
            axis_title_z:Optional[str]=None,
            show:bool=False):
    '''
    Creates a surface plot using a function. Returns the figure for that plot.
    Note that if calling this from jupyter notebooks and not capturing the output
    it will appear on screen as though `.show()` has been called
    x_value: A numpy array of x values
    y_value: A numpy array of y values
    calc_z: A function to calculate z, given an x and a y value
    title: Plot title
    axis_title_x: Title for the x axis
    axis_title_y: Title for the y axis
    axis_title_z: Title for the z axis
    show:   appears on screen. NB that this is not needed if this is called from a
            notebook and the output is not captured 
    '''

    # Check arguments
    assert len(x_values.shape) == 1, "Provide x_values as 1D"
    assert len(y_values.shape) == 1, "Provide y_values as 1D"


    # Calculate cost for a range of intercepts and slopes
    # intercepts = np.linspace(-100,-70,10)
    # slopes = np.linspace([0.060],[0.07],10, axis=1)
    z = numpy.zeros((y_values.shape[0], x_values.shape[0]))
    for i_x in range(x_values.shape[0]):
        for i_y in range(y_values.shape[0]):
            z[i_y, i_x] = calc_z(x_values[i_x], y_values[i_y])
            
    # Create a graph of cost
    fig = graph_objects.Figure(data=[graph_objects.Surface(x=x_values, y=y_values, z=z)])
    fig.update_layout(title=title, 
                      scene_xaxis_title=axis_title_x, 
                      scene_yaxis_title=axis_title_y, 
                      scene_zaxis_title=axis_title_z)

    # Show the plot, if requested
    if show:
        fig.show()
    
    # return the figure
    return fig