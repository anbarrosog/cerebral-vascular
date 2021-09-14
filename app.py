######## LIBRARIES ########
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pydicom
import base64
import io
import cv2
from skimage import data
import json
from PIL import Image
from sklearn import mixture
from sklearn.mixture import GaussianMixture as GMM
import skimage as sc
import skimage
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
from skimage import data, morphology
import matplotlib.pyplot as plt
from skimage.util import invert
from scipy import misc,ndimage
import scipy.ndimage as ndi
from mahotas.morph import hitmiss as hit_or_miss
from skimage.measure import label, regionprops, regionprops_table
import matplotlib.patches as mpatches
import math
import pandas as pd
from scipy.cluster.vq import vq
from itertools import product
from skan import skeleton_to_csgraph
from dash_table import DataTable, FormatTemplate

DEBUG = False

# Set up the app
external_stylesheets = [dbc.themes.BOOTSTRAP, "assets/object_properties_style.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server


######## INITIALIZATION ########

iarray = None
percentage = FormatTemplate.percentage(2)


######ALL FUNCTIONS DEFINED######
#load image
def xor(a, b):
    "Same as a ^ b."
    return a ^ b

def load_default_dicom(
        images='assets/dicomdirectory/imatgeprova.dcm',
):
    fig = px.imshow(pydicom.dcmread(images).pixel_array, color_continuous_scale='gray')
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        dragmode="drawopenpath",
        margin=dict(l=0, r=0, b=0, t=0, pad=4),
    )
    return fig

def dicom_to_fig(
       contents,
):
    content_type, content_string = contents.split(',')
    dicom = base64.b64decode(content_string)
    ds = pydicom.dcmread(io.BytesIO(dicom))
    iarray = ds.pixel_array

    fig = px.imshow(iarray, color_continuous_scale='gray')
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        dragmode="drawopenpath",
        margin=dict(l=0, r=0, b=0, t=0, pad=4),
    )
    return fig

#fuction set to find branchpoints in the deep veins skel
def find_branch_points(skel):
    X=[]
    #cross X
    X0 = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]])
    X1 = np.array([[1, 0, 1],
                   [0, 1, 0],
                   [1, 0, 1]])
    X.append(X0)
    X.append(X1)
    #T like
    T=[]
    #T0 contains X0
    T0=np.array([[2, 1, 2],
                 [1, 1, 1],
                 [2, 2, 2]])

    T1=np.array([[1, 2, 1],
                 [2, 1, 2],
                 [1, 2, 2]])  # contains X1

    T2=np.array([[2, 1, 2],
                 [1, 1, 2],
                 [2, 1, 2]])

    T3=np.array([[1, 2, 2],
                 [2, 1, 2],
                 [1, 2, 1]])

    T4=np.array([[2, 2, 2],
                 [1, 1, 1],
                 [2, 1, 2]])

    T5=np.array([[2, 2, 1],
                 [2, 1, 2],
                 [1, 2, 1]])

    T6=np.array([[2, 1, 2],
                 [2, 1, 1],
                 [2, 1, 2]])

    T7=np.array([[1, 2, 1],
                 [2, 1, 2],
                 [2, 2, 1]])
    T.append(T0)
    T.append(T1)
    T.append(T2)
    T.append(T3)
    T.append(T4)
    T.append(T5)
    T.append(T6)
    T.append(T7)
    #Y like
    Y=[]
    Y0=np.array([[1, 0, 1],
                 [0, 1, 0],
                 [2, 1, 2]])

    Y1=np.array([[0, 1, 0],
                 [1, 1, 2],
                 [0, 2, 1]])

    Y2=np.array([[1, 0, 2],
                 [0, 1, 1],
                 [1, 0, 2]])

    Y2=np.array([[1, 0, 2],
                 [0, 1, 1],
                 [1, 0, 2]])

    Y3=np.array([[0, 2, 1],
                 [1, 1, 2],
                 [0, 1, 0]])

    Y4=np.array([[2, 1, 2],
                 [0, 1, 0],
                 [1, 0, 1]])
    Y5=np.rot90(Y3)
    Y6 = np.rot90(Y4)
    Y7 = np.rot90(Y5)
    Y.append(Y0)
    Y.append(Y1)
    Y.append(Y2)
    Y.append(Y3)
    Y.append(Y4)
    Y.append(Y5)
    Y.append(Y6)
    Y.append(Y7)

    bp = np.zeros(skel.shape, dtype=int)
    for x in X:
        bp = bp + hit_or_miss(skel,x)
    for y in Y:
        bp = bp + hit_or_miss(skel,y)
    for t in T:
        bp = bp + hit_or_miss(skel,t)

    return bp


#function set to find endpoints in deep veins skel
def find_end_points(skel):
    endpoint1=np.array([[0, 0, 0],
                        [0, 1, 0],
                        [2, 1, 2]])

    endpoint2=np.array([[0, 0, 0],
                        [0, 1, 2],
                        [0, 2, 1]])

    endpoint3=np.array([[0, 0, 2],
                        [0, 1, 1],
                        [0, 0, 2]])

    endpoint4=np.array([[0, 2, 1],
                        [0, 1, 2],
                        [0, 0, 0]])

    endpoint5=np.array([[2, 1, 2],
                        [0, 1, 0],
                        [0, 0, 0]])

    endpoint6=np.array([[1, 2, 0],
                        [2, 1, 0],
                        [0, 0, 0]])

    endpoint7=np.array([[2, 0, 0],
                        [1, 1, 0],
                        [2, 0, 0]])

    endpoint8=np.array([[0, 0, 0],
                        [2, 1, 0],
                        [1, 2, 0]])

    ep1=hit_or_miss(skel,endpoint1)
    ep2=hit_or_miss(skel,endpoint2)
    ep3=hit_or_miss(skel,endpoint3)
    ep4=hit_or_miss(skel,endpoint4)
    ep5=hit_or_miss(skel,endpoint5)
    ep6=hit_or_miss(skel,endpoint6)
    ep7=hit_or_miss(skel,endpoint7)
    ep8=hit_or_miss(skel,endpoint8)
    ep = ep1+ep2+ep3+ep4+ep5+ep6+ep7+ep8
    return ep

#function to divide the image into two hemispheres at the height of the landmark
def imCrop(x):
    height,width,depth = x.shape
    skel=cv2.imread('assets/skeleton.jpg')
    branch_pts1 = find_branch_points(skel)
    pixels = np.asarray(branch_pts1)
    coords = np.column_stack(np.where(pixels == 1))
    landmark=coords[2]
    return [x[: , :landmark[1]] , x[:, (width-landmark[1]):]]

## function that converts a colour image to a greyscale image###
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

#####define the initial image to be plotted

img = cv2.imread('assets/gray.jpg',0)
img_2 = np.zeros(img.shape)
fig_2 = px.imshow(img_2, binary_string = True)
fig_2.update_layout(
    margin = dict(l=0, r=0, b=0, t=0, pad=4),
    uirevision = True

)

##### We create all dashboard buttons##
run_segmentation= dbc.Button(
    'Run Segmentation',
    id = 'button_run_segmentation',
    outline = True,
    color ='primary',
)

Button_run_segmentation = html.Div(
    [
        dbc.Row(
            run_segmentation,
        )
    ]
)

run_intensity= dbc.Button(
    'Run Intensity',
    id = 'button_run_intensity',
    outline = True,
    color ='primary',
)

Button_run_intensity = html.Div(
    [
        dbc.Row(
            run_intensity
        )
    ]
)

run_skel= dbc.Button(
    'Run Skel',
    id = 'button_run_skel',
    outline = True,
    color ='primary',
)

Button_run_skel = html.Div(
    [
        dbc.Row(
            run_skel
        )
    ]
)

run_bpep= dbc.Button(
    'Run Detection',
    id = 'button_run_bpep',
    outline = True,
    color ='primary',
)

Button_run_bpep = html.Div(
    [
        dbc.Row(
            run_bpep
        )
    ]
)

run_land= dbc.Button(
    'Run LandMark',
    id = 'button_run_land',
    outline = True,
    color ='primary',
)

Button_run_land = html.Div(
    [
        dbc.Row(
            run_land
        )
    ]
)

run_div= dbc.Button(
    'Run Division',
    id = 'button_run_div',
    outline = True,
    color ='primary',
)

Button_run_div = html.Div(
    [
        dbc.Row(
            run_div
        )
    ]
)

run_bio= dbc.Button(
    'Run Biomarkers',
    id = 'button_run_bio',
    outline = True,
    color ='primary',
)

Button_run_bio = html.Div(
    [
        dbc.Row(
            run_bio
        )
    ]
)

# ------------- Cards definition ---------------------------------------------------

##loads and displays the selected dicom file

carrega = dbc.Card(id='carrega', children=
        [
            dbc.CardHeader(html.H4(dbc.Badge("Brain's MRI image",className="m1-1"))),
            dbc.CardBody(
            [
                dcc.Upload(id='upload-button', children=html.Button('Upload DICOM', style={'color':'white', 'backgroundColor':'blue','border':'1.5px blue solid'}), multiple=True),
                html.Div(id='image-original'),
                dcc.Graph(
                                        id="graph",
                                        figure=load_default_dicom(),
                                        config={
                                            "modeBarButtonsToAdd": [
                                                "drawrect",
                                                "drawopenpath",
                                                "drawclosedpath",
                                                "eraseshape"]}),
                #dcc.Markdown("Characteristics of shapes"),
                html.Pre(id="annotations-data", style={'display': 'none'}),
            ]
            ),

            dbc.CardFooter(
            [
                html.H6(
                    [
                        "First step:\n\nDraw a rectangle, with the plotly tool of the panel, in on the region of interest where the ",
                        html.Span(
                            "deep veins of the brain are located",
                            id="tooltip-target-1",
                            className="tooltip-target",
                            style={"textDecoration": "underline", "cursor": "pointer"},
                        ),
                        ".",
                    ]
                ),
                dbc.Tooltip(
                    "Focus only on the part of the image where most of the internal veins are located.",
                    target="tooltip-target-1",
                ),
                dbc.Button("STEP 1", id="button_run_segmentation", color="primary", block=True)
            ]
            ),
        ],
)

####Plots the segmented image from the previous panel. The area that has been left inside the drawn rectangle

segmented = dbc.Card(
    id = 'segmented',
    children = [
        dbc.CardHeader(html.H4(dbc.Badge("Selected region",className="m1-1"))),
        dbc.CardBody(
            children = [
                dcc.Graph(
                    id = 'figure_segmented',
                    figure = fig_2,
                    config = {
                        'scrollZoom' : True,
                    }
                ),
            ]
        ),
            dbc.CardFooter(
            [
                html.H6(
                    [
                        "Second step:\n\nRegion selected (ROI) where most ",
                        html.Span(
                            "deep veins of the brain are located",
                            id="tooltip-target-2",
                            className="tooltip-target",
                            style={"textDecoration": "underline", "cursor": "pointer"},
                        ),
                        ".",
                    ]
                ),
                dbc.Tooltip(
                    "In case you have not selected the desired part use the 'erase active shape' tool from the previous panel and redraw the rectangle",
                    target="tooltip-target-2",
                ),
                dbc.Button("STEP 2", id="button_run_intensity", color="primary", block=True)
            ]
            ),
    ],
    # style={"width": "18rem"},
),

####Shows the binarised image after applying the Gaussian intensity model.

intensity = dbc.Card(
    id = 'intensity',
    children = [
        dbc.CardHeader(html.H4(dbc.Badge("Binarised image according to intensity model",className="m1-1"))),
        dbc.CardBody(
            children = [
                dcc.Graph(
                    id = 'figure_intensity',
                    figure = fig_2,
                    config = {
                        'scrollZoom' : True,
                    }
                ),
            ]
        ),
            dbc.CardFooter(
            [
                html.H6(
                    [
                        "Third step:\n\nThe image shows a binary image once",
                        html.Span(
                            " the Gaussian Mixture Model",
                            id="tooltip-target-3",
                            className="tooltip-target",
                            style={"textDecoration": "underline", "cursor": "pointer"},
                        ),
                        " , based on instensity, has been applied.",
                    ]
                ),
                dbc.Tooltip(
                    "Using this segmentation model we can distinguish areas of the image according to the intensity they present",
                    target="tooltip-target-3",
                ),
                dbc.Button("STEP 3", id="button_run_skel", color="primary", block=True)
            ]
            ),
    ],
    # style={"width": "18rem"},
),

#### The skeleton of the internal veins of the brain is visualised###

skel = dbc.Card(
    id = 'skel',
    children = [
        dbc.CardHeader(html.H4(dbc.Badge("Deep veins skeleton",className="m1-1"))),
        dbc.CardBody(
            children = [
                dcc.Graph(
                    id = 'figure_skel',
                    figure = fig_2,
                    config = {
                        'scrollZoom' : True,
                    }
                ),
            ]
        ),
            dbc.CardFooter(
            [
                html.H6(
                    [
                        "Fourth step:\n\nThe image shows the skeleton",
                        html.Span(
                            " of the deep veins of the brain",
                            id="tooltip-target-4",
                            className="tooltip-target",
                            style={"textDecoration": "underline", "cursor": "pointer"},
                        ),
                        ".",
                    ]
                ),
                dbc.Tooltip(
                    "At this point we have obtained the skeleton of the veins and we can extract a series of biomarkers that will allow us to evaluate the symmetry.",
                    target="tooltip-target-4",
                ),
                dbc.Button("STEP 4", id="button_run_bpep", color="primary", block=True)
            ]
            ),
    ],
    # style={"width": "18rem"},
),

### Mapping of branch points and end points

bpep = dbc.Card(
    id = 'bpep',
    children = [
        dbc.CardHeader(html.H4(dbc.Badge("Deep veins branchpoints and endpoints",className="m1-1"))),
        dbc.CardBody(
            children = [
                dcc.Graph(
                    id = 'figure_bpep',
                    figure = fig_2,
                    config = {
                        'scrollZoom' : True,
                    }
                ),
            ]
        ),
            dbc.CardFooter(
            [
                html.H6(
                    [
                        "Fifth step:\n\nThe image shows the skeleton ",
                        html.Span(
                            "with all branch points and end points marked",
                            id="tooltip-target-5",
                            className="tooltip-target",
                            style={"textDecoration": "underline", "cursor": "pointer"},
                        ),
                        ".",
                    ]
                ),
                dbc.Tooltip(
                    "These parameters allows us to operate and obtain the centroid of the image to separate the two cerebral hemispheres at the height of the vein of galenus",
                    target="tooltip-target-5",
                ),
                dbc.Button("STEP 5", id="button_run_land", color="primary", block=True)
            ]
            ),
    ],
    # style={"width": "18rem"},
),

#### Detected landmark fitted with a line marking the hemispheric division
land = dbc.Card(
    id = 'land',
    children = [
        dbc.CardHeader(html.H4(dbc.Badge("Landmark setting (Galen vein)",className="m1-1"))),
        dbc.CardBody(
            children = [
                dcc.Graph(
                    id = 'figure_land',
                    figure = fig_2,
                    config = {
                        'scrollZoom' : True,
                    }
                ),
            ]
        ),
        dbc.CardFooter(
        [
            html.H6(
                [
                    "Sixth step:\n\nThe image shows the initial segmented image with the division lanmark of the two hemispheres",
                    html.Span(
                        " aligned to the centroid and at the level of the galenic vein",
                        id="tooltip-target-6",
                        className="tooltip-target",
                        style={"textDecoration": "underline", "cursor": "pointer"},
                    ),
                    ".",
                ]
            ),
            dbc.Tooltip(
                "This landmark is not rigorous and sometimes it may not be located in the center of the image, in any case, if so, restart the process by drawing the rectangle again in the first panel",
                target="tooltip-target-6",
            ),
            dbc.Button("STEP 6", id="button_run_div", color="primary", block=True)
        ]
        ),
    ],
    # style={"width": "18rem"},
),

### visualization of the skeleton of both hemispheres

division = dbc.Card(
    id = 'division',
    children = [
        dbc.CardHeader(html.H4(dbc.Badge("Cerebral skeleton hemispheres division ",className="m1-1"))),
        dbc.CardBody(
            children = [
                dcc.Graph(
                    id = 'figure_div',
                    figure = fig_2,
                    config = {
                        'scrollZoom' : True,
                    }
                ),
            ]
        ),
        dbc.CardFooter(
        [
            html.H6(
                [
                    "Seventh step:\n\nThe image shows the skeleton of",
                    html.Span(
                        " the right hemisphere and left hemisphere",
                        id="tooltip-target-7",
                        className="tooltip-target",
                        style={"textDecoration": "underline", "cursor": "pointer"},
                    ),
                    ".",
                ]
            ),
            dbc.Tooltip(
                "Once we have obtained the skeleton of both hemispheres, we can extract a series of parameters that allow us to evaluate symmetry.",
                target="tooltip-target-7",
            ),
            dbc.Button("STEP 7", id="button_run_bio", color="primary", block=True)
        ]
        ),

    ],
    # style={"width": "18rem"},
),
### table format panel containing all the calculated percentages of symmetrical comparision
biomarkers = dbc.Card(
    id = 'biomarkers',
    children = [
                dbc.CardHeader(html.H4(dbc.Badge("Ratios of biomarkers",className="m1-1"))),
                dbc.CardBody(html.Div([
                dash_table.DataTable(
                    id='loading-table',
                    columns = [
                                dict(id='parameter', name='Parameter'),
                                dict(id='right_hemisphere', name='Right Hemisphere', type='numeric', format=percentage),
                                dict(id='left_hemisphere', name='Left Hemisphere', type='numeric', format=percentage)
                            ],
                            data = [
                                    dict(parameter='Branch Points', right_hemisphere=0, left_hemisphere=0),
                                    dict(parameter='End Points',  right_hemisphere=0, left_hemisphere=0),
                                    dict(parameter='Medium Length',  right_hemisphere=0, left_hemisphere=0),
                                    dict(parameter='Total Length',  right_hemisphere=0, left_hemisphere=0),
                                    dict(parameter='Specular Overlap',  right_hemisphere=0, left_hemisphere=0)
                                ],
                    editable=True,
                    style_data_conditional=[
                        {
                            'if': {
                                'filter_query': '{right_hemisphere} > 0.4 && {right_hemisphere} < 0.6',
                                'column_id': 'right_hemisphere'
                            },
                            'backgroundColor': '#3D9970',
                            'fontWeight': 'bold'
                        },
                        {
                            'if': {
                                'filter_query': '{left_hemisphere} > 0.4 && {left_hemisphere} < 0.6',
                                'column_id': 'left_hemisphere'
                            },
                            'backgroundColor': '#3D9970',
                            'fontWeight': 'bold'
                        },
                        {
                            'if': {
                                'filter_query': '{right_hemisphere} < 0.4',
                                'column_id': 'right_hemisphere'
                            },
                            'backgroundColor': 'tomato',
                            'fontWeight': 'bold'
                        },
                        {
                            'if': {
                                'filter_query': '{left_hemisphere} < 0.4',
                                'column_id': 'left_hemisphere'
                            },
                            'backgroundColor': 'tomato',
                            'fontWeight': 'bold'
                        },
                        {
                            'if': {
                                'filter_query': '{right_hemisphere} > 0.6',
                                'column_id': 'right_hemisphere'
                            },
                            'backgroundColor': 'tomato',
                            'fontWeight': 'bold'
                        },
                        {
                            'if': {
                                'filter_query': '{left_hemisphere} >0.6',
                                'column_id': 'left_hemisphere'
                            },
                            'backgroundColor': 'tomato',
                            'fontWeight': 'bold'
                        },
                        ],
                ),
                ]),
        ),
        dbc.CardFooter(
        [
            html.H6(
                [
                    "Eighth step:\n\nThe table shows the biomarkers extracted and calculated the ratio on a percentage basis. In case",
                    html.Span(
                        " of symmetry the biomarkers should be distributed 50-50 in both hemispheres",
                        id="tooltip-target-8",
                        className="tooltip-target",
                        style={"textDecoration": "underline", "cursor": "pointer"},
                    ),
                    ".",
                ]
            ),
            dbc.Tooltip(
                "In case three or more ratios of the calculated biomarkers are in green cells, the RMI of the pediatric patient's brain can be considered as symmetric.",
                target="tooltip-target-8",
            )
        ]
        ),
    ],
    # style={"width": "18rem"},
),

#------------------------------ Define Modal; separate windows within an application-------------------------

with open("assets/modal.md", "r") as f:
    howto_md = f.read()

modal_overlay = dbc.Modal(
    [
        dbc.ModalBody(html.Div([dcc.Markdown(howto_md)], id="howto-md")),
        dbc.ModalFooter(dbc.Button("Close", id="howto-close", className="howto-bn")),
    ],
    id="modal",
    size="lg",
)

###### Buttons of the modal bar###

button_gh = dbc.Button(
    "Learn more",
    id="howto-open",
    outline=True,
    color="secondary",
    # Turn off lowercase transformation for class .button in stylesheet
    style={"textTransform": "none"},
)

button_howto = dbc.Button(
    "View Code on github",
    outline=True,
    color="primary",
    href="https://github.com/anbarrosog/vascular-cerebral-patterns-app",
    id="gh-link",
    style={"text-transform": "none"},
)

#-----------------------Define Header Layout-------------------------------------#

header = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.A(
                            html.Img(
                                src=app.get_asset_url("dash-logo-new.png"),
                                height="30px",
                            ),
                            href="https://plotly.com/dash/",
                        )
                    ),
                    dbc.Col(dbc.NavbarBrand("Cerebral Vascular App")),
                    modal_overlay,
                ],
                align="center",
            ),
            dbc.Row(
                dbc.Col(
                    [
                        dbc.NavbarToggler(id="navbar-toggler"),
                        dbc.Collapse(
                            dbc.Nav(
                                [dbc.NavItem(button_howto), dbc.NavItem(button_gh)],
                                className="ml-auto",
                                navbar=True,
                            ),
                            id="navbar-collapse",
                            navbar=True,
                        ),
                    ]
                ),
                align="center",
            ),
        ],
        fluid=True,
    ),
    color="dark",
    dark=True,
)
####################################### LAYOUT ########################################

####---------Define APP Layout-------------

app.layout = html.Div(
    [
            header,
            dbc.Container(
                children=[
                    dbc.Row([dbc.Col(carrega),dbc.Col(segmented), dbc.Col(intensity)]),
                    dbc.Row([dbc.Col(skel),dbc.Col(bpep),dbc.Col(land)]),
                    dbc.Row([dbc.Col(division, md=4),dbc.Col(biomarkers, md=4)]),
                ],
                fluid= True,
               ),
     ]
 )


####################################### CALLBACK ########################################

# ------------- Define App Interactivity ---------------------------------------------------

@app.callback(Output('graph', 'figure'),
              Input('upload-button', 'contents'),
              State('upload-button', 'filename'))


def load_dicom(list_of_contents, list_of_names):
    if list_of_contents is not None:
        dicom_image=dicom_to_fig(list_of_contents[0])
        return dicom_image
    else:
        return dash.no_update

# we use a callback to toggle the collapse on small screens

@app.callback(
    Output("modal", "is_open"),
    [Input("howto-open", "n_clicks"), Input("howto-close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


# we use a callback to toggle the collapse on small screens
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

### Coordinates and adjust image by the region defined by the rectangle.
@app.callback(
    Output("annotations-data", "children"),
    Input("graph", "relayoutData"),
    prevent_initial_call=True,
)

def on_new_annotation(relayout_data):
    if "shapes" in relayout_data:
        info=relayout_data["shapes"]
        x0= info[0]["x0"]
        x1= info[0]["x1"]
        y0= info[0]["y0"]
        y1= info[0]["y1"]
        n = Image.open('assets/gray.jpg')
        cropped_img = n.crop((x0,y0,x1,y1))
        cropped_img.save("assets/crop1.jpg")
        #cv2.imwrite("assets/crop1.jpg", crop_img)
        #test_jpg = 'crop1.jpg'
        #test_base64 = base64.b64encode(open(test_jpg, 'rb').read()).decode('ascii')
        #plt.imshow(crop_img, cmap='gray', interpolation='nearest'); plt.show()
        return json.dumps(relayout_data["shapes"], indent=2)
    else:
        return dash.no_update


@app.callback(
    Output('figure_segmented', 'figure'),
    [
        Input('graph', 'figure'),
        Input('button_run_segmentation', 'n_clicks'),
    ]
)
def figure_segmented(fig_input, n_clicks):

    if n_clicks is None:
        figure_white = px.imshow(img_2, binary_string = True)
        figure_white.update_xaxes(showticklabels=False)
        figure_white.update_yaxes(showticklabels=False)
        figure_white.update_layout(
            margin = dict(l=0, r=0, b=0, t=0, pad=4),
            uirevision = True
        ),
        return figure_white

    else:
        img_cropped= cv2.imread('assets/crop1.jpg',0)
        figure_seg= px.imshow(img_cropped, binary_string = True)
        figure_seg.update_xaxes(showticklabels=False)
        figure_seg.update_yaxes(showticklabels=False)
        figure_seg.update_layout(
            margin = dict(l=0, r=0, b=0, t=0, pad=4),
            uirevision = True
        ),

        return figure_seg

@app.callback(
    Output('figure_intensity', 'figure'),
    [
        Input('graph', 'figure'),
        Input('button_run_intensity', 'n_clicks'),
    ]
)


def figure_intensity(fig_input1, n_clicks1):

    if n_clicks1 is None:
        figure_white = px.imshow(img_2, binary_string = True)
        figure_white.update_xaxes(showticklabels=False)
        figure_white.update_yaxes(showticklabels=False)
        figure_white.update_layout(
            margin = dict(l=0, r=0, b=0, t=0, pad=4),
            uirevision = True
        ),
        return figure_white

    else:
        img=cv2.imread('assets/crop1.jpg')
        img2 = img.reshape((-1,3))
        gmm_model = GMM(n_components=6, covariance_type='tied', random_state=3).fit(img2)  #tied works better than full
        gmm_labels = gmm_model.predict(img2)
        original_shape = img.shape
        intensity = gmm_labels.reshape(original_shape[0], original_shape[1])
        cv2.imwrite("assets/segmented.jpg", intensity)
        image_int = cv2.cvtColor(cv2.imread('assets/segmented.jpg'), cv2.COLOR_BGR2GRAY)
        th, im_gray_th_otsu = cv2.threshold(image_int, 72, 255, cv2.THRESH_OTSU)
        cv2.imwrite('assets/binary.jpg', im_gray_th_otsu)
        figure_binary= px.imshow(cv2.imread('assets/binary.jpg'))
        figure_binary.update_xaxes(showticklabels=False)
        figure_binary.update_yaxes(showticklabels=False)
        figure_binary.update_layout(
            margin = dict(l=0, r=0, b=0, t=0, pad=4),
            uirevision = True
        ),

        return figure_binary

@app.callback(
    Output('figure_skel', 'figure'),
    [
        Input('graph', 'figure'),
        Input('button_run_skel', 'n_clicks'),
    ]
)


def figure_skel(fig_input2, n_clicks2):

    if n_clicks2 is None:
        figure_white = px.imshow(img_2, binary_string = True)
        figure_white.update_xaxes(showticklabels=False)
        figure_white.update_yaxes(showticklabels=False)
        figure_white.update_layout(
            margin = dict(l=0, r=0, b=0, t=0, pad=4),
            uirevision = True
        ),
        return figure_white

    else:
        image_bin = cv2.cvtColor(cv2.imread('assets/binary.jpg'), cv2.COLOR_BGR2GRAY)
        th1, im_gray_th_otsu1 = cv2.threshold(image_bin, 128, 255, cv2.THRESH_BINARY)
        imfill= ndimage.binary_fill_holes(im_gray_th_otsu1)
        dil_img = ndimage.binary_dilation(imfill)
        open_img = ndimage.binary_opening(dil_img)
        close_img = ndimage.binary_closing(open_img)
        clg = close_img.astype(np.int)
        skeleton = skeletonize(clg)
        cv2.imwrite("assets/skeleton.jpg", skeleton*255)
        figure_skel= px.imshow(cv2.imread('assets/skeleton.jpg'))
        figure_skel.update_xaxes(showticklabels=False)
        figure_skel.update_yaxes(showticklabels=False)
        figure_skel.update_layout(
            margin = dict(l=0, r=0, b=0, t=0, pad=4),
            uirevision = True
        ),

        return figure_skel

@app.callback(
    Output('figure_bpep', 'figure'),
    [
        Input('graph', 'figure'),
        Input('button_run_bpep', 'n_clicks'),
    ]
)


def figure_bpep(fig_input3, n_clicks3):

    if n_clicks3 is None:
        figure_white = px.imshow(img_2, binary_string = True)
        figure_white.update_xaxes(showticklabels=False)
        figure_white.update_yaxes(showticklabels=False)
        figure_white.update_layout(
            margin = dict(l=0, r=0, b=0, t=0, pad=4),
            uirevision = True
        ),
        return figure_white

    else:
        skeleton=cv2.imread('assets/skeleton.jpg')
        branch_pts = find_branch_points(skeleton)
        end_pts = find_end_points(skeleton)
        detected= end_pts+branch_pts+skeleton
        cv2.imwrite("assets/detected.jpg", detected*255)
        figure_bpep= px.imshow(cv2.imread('assets/detected.jpg'))
        figure_bpep.update_xaxes(showticklabels=False)
        figure_bpep.update_yaxes(showticklabels=False)
        figure_bpep.update_layout(
            margin = dict(l=0, r=0, b=0, t=0, pad=4),
            uirevision = True
        ),

        return figure_bpep


@app.callback(
    Output('figure_land', 'figure'),
    [
        Input('graph', 'figure'),
        Input('button_run_land', 'n_clicks'),
    ]
)


def figure_land(fig_input5, n_clicks5):

    if n_clicks5 is None:
        figure_white = px.imshow(img_2, binary_string = True)
        figure_white.update_xaxes(showticklabels=False)
        figure_white.update_yaxes(showticklabels=False)
        figure_white.update_layout(
            margin = dict(l=0, r=0, b=0, t=0, pad=4),
            uirevision = True
        ),
        return figure_white

    else:
        skel=cv2.imread('assets/skeleton.jpg')
        branch_pts1 = find_branch_points(skel)
        pixels = np.asarray(branch_pts1)
        coords = np.column_stack(np.where(pixels == 1))
        landmark=coords[2]
        props = regionprops(skel)
        centroid=props[0]['Centroid']
        image = cv2.imread ("assets/crop1.jpg")
        height = image.shape[0]
        width = image.shape[1]
        cv2.line(image, (landmark[1],0), (landmark[1],height), (255,0,0), 1)
        cv2.imwrite("assets/image.jpg", image*255)
        figure_land= px.imshow(cv2.imread('assets/image.jpg'))
        figure_land.update_xaxes(showticklabels=False)
        figure_land.update_yaxes(showticklabels=False)
        figure_land.update_layout(
            margin = dict(l=0, r=0, b=0, t=0, pad=4),
            uirevision = True
        ),

        return figure_land

@app.callback(
    Output('figure_div', 'figure'),
    [
        Input('graph', 'figure'),
        Input('button_run_div', 'n_clicks'),
    ]
)



def figure_div(fig_input6, n_clicks6):

    if n_clicks6 is None:
        figure_white = px.imshow(img_2, binary_string = True)
        figure_white.update_xaxes(showticklabels=False)
        figure_white.update_yaxes(showticklabels=False)
        figure_white.update_layout(
            margin = dict(l=0, r=0, b=0, t=0, pad=4),
            uirevision = True
        ),
        return figure_white

    else:
        divide=cv2.imread('assets/image.jpg')
        firsthem=imCrop(divide)[0]
        secondhem=imCrop(divide)[1]
        cv2.imwrite("assets/firsthem.jpg", firsthem)
        cv2.imwrite("assets/secondhem.jpg", secondhem)
        grayhem1=rgb2gray(firsthem)
        grayhem2=rgb2gray(secondhem)


        thresh1 = threshold_otsu(grayhem1)
        binary1 = grayhem1 > thresh1

        thresh2 = threshold_otsu(grayhem2)
        binary2 = grayhem2 > thresh2

        imfillhem1= ndimage.binary_fill_holes(binary1)
        imfillhem2= ndimage.binary_fill_holes(binary2)

        open_img1 = ndimage.binary_opening(imfillhem1)
        close_img1 = ndimage.binary_closing(open_img1)
        clg1 = close_img1.astype(np.int)

        open_img2 = ndimage.binary_opening(imfillhem2)
        close_img2 = ndimage.binary_closing(open_img2)
        clg2 = close_img2.astype(np.int)

        skeletonhem1 = skeletonize(clg1)
        skeletonhem2 = skeletonize(clg2)

        cv2.imwrite("assets/skel1.jpg", skeletonhem1*255)
        cv2.imwrite("assets/skel2.jpg", skeletonhem2*255)
        figure_div= make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.25)
        figure_div.add_trace(px.imshow(cv2.imread('assets/skel1.jpg')).data[0], 1, 1)
        figure_div.add_trace(px.imshow(cv2.imread('assets/skel2.jpg')).data[0], 1, 2)
        figure_div.update_xaxes(showticklabels=False)
        figure_div.update_yaxes(showticklabels=False)
        figure_div.update_layout(
            margin = dict(l=0, r=0, b=0, t=0, pad=4),
            uirevision = True
        ),

        return figure_div


@app.callback(
    Output('loading-table', 'data'),
    Input('button_run_bio', 'n_clicks'),
    State('loading-table', 'data')
)


def table_bio(n_clicks7, table_param):

    if n_clicks7 is None:
        return table_param
    else:
        #First biomarker: finding branch points
        hem1=cv2.imread('assets/skel1.jpg')
        hem2=cv2.imread('assets/skel2.jpg')
        branchpointshem1= find_branch_points(hem1)
        branchpointshem2= find_branch_points(hem2)

        number_of_white_pix_hem1_branchpts = np.sum (branchpointshem1 == 1)
        number_of_white_pix_hem2_branchpts = np.sum (branchpointshem2 == 1)
        calcperbranchhem1=((number_of_white_pix_hem1_branchpts)/(number_of_white_pix_hem1_branchpts+number_of_white_pix_hem2_branchpts))
        calcperbranchhem2=1-calcperbranchhem1

        #Second biomarker: finding end points
        endpointshem1= find_end_points(hem1)
        endpointshem2= find_end_points(hem2)

        number_of_white_pix_hem1_endpts = np.sum (endpointshem1 == 1)
        number_of_white_pix_hem2_endpts = np.sum (endpointshem2 == 1)
        calcperendhem1=((number_of_white_pix_hem1_endpts)/(number_of_white_pix_hem1_endpts+number_of_white_pix_hem2_endpts))
        calcperendhem2=1-calcperendhem1

        #Third biomarker: calculate the medium length of all hemispheric veins
        endbranch1= np.asarray(branchpointshem1 + endpointshem1 + hem1)
        endbranch2= np.asarray(branchpointshem2 + endpointshem2 + hem2)

        pixel_graph1, coordinates1, degrees1 = skeleton_to_csgraph(endbranch1)
        pixel_graph2, coordinates2, degrees2 = skeleton_to_csgraph(endbranch2)


        dif_long_med=(sum((coordinates1[1])/100)/2) #if the ratio we ideally calculated were 1 this would mean that the two hemispheres have the same average length


        #Fourth biomarker: calculate the total length of all hemispheric veins

        dif_long_tot2=sum(coordinates2[1])/100

        #Fifth biomarker: correlating the image of one hemisphere with the mirror image of the other
        esp_hem2 = cv2.flip(cv2.imread('assets/skel2.jpg'),1)
        imfill2= ndimage.binary_fill_holes(esp_hem2)
        dil_img2 = ndimage.binary_dilation(imfill2)
        dil2 = dil_img2.astype(np.int)

        imfill1= ndimage.binary_fill_holes(cv2.imread('assets/skel1.jpg'))
        dil_img1 = ndimage.binary_dilation(imfill1)
        dil1 = dil_img1.astype(np.int)


        Isim= xor(dil1, dil2)

        ratiNC=((np.sum (Isim == 1))/(np.sum (dil1 == 1)+np.sum (dil2 == 1)))

        data = [
                dict(parameter='Branch Points',  right_hemisphere=calcperbranchhem1, left_hemisphere=calcperbranchhem2),
                dict(parameter='End Points', right_hemisphere=calcperendhem1, left_hemisphere=calcperendhem2),
                dict(parameter='Medium Length', right_hemisphere=dif_long_med, left_hemisphere=1-dif_long_med),
                dict(parameter='Total Length', right_hemisphere=1-dif_long_tot2, left_hemisphere=dif_long_tot2),
                dict(parameter='Specular Overlap', right_hemisphere=ratiNC, left_hemisphere=1-ratiNC)
            ]
        return data

if __name__ == "__main__":
    app.run_server()

##### Replace the lines of run server with the following in case you want to deploy the application in docker
'''
if __name__ == '__main__':
        app.run_server(host='0.0.0.0', port=8050, debug=DEBUG)
'''
