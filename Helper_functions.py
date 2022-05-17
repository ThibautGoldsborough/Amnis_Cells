#Adapted from https://colab.research.google.com/drive/1nSGnnlt_dwNFDNxCMIuDxk-AsujIdncP?usp=sharing
import io
import base64
from base64 import b64encode
from jupyter_dash import JupyterDash
from dash import dcc, html, Input, Output, no_update
import plotly.graph_objects as go
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt


def load_dict(outpath,cell_names,image_dim):
    image_dict={}
    for cell_name in cell_names:
        image_dict[cell_name]={}
    #Find Channels
    names=[]
    for entry in os.listdir(outpath): #Read all files
        if os.path.isfile(os.path.join(outpath, entry)):
            if entry!='image_ID.npy':
                names.append(entry)
    channels=[name[:-4] for name in names if name[-4:]=='.npy']
    print("Channels found:",channels)
    data_dict={}
    for channel in channels:
        data_dict[channel]=np.load(outpath+"\\"+channel+'.npy')
    #Break up array
    for channel in data_dict:
        dims=data_dict[channel].shape
        n=dims[0]//image_dim
        l=dims[1]//image_dim
        index=0
        for i in range(n):
            for j in range(l):
                img=data_dict[channel][i*image_dim:i*image_dim+image_dim,j*image_dim:j*image_dim+image_dim]
                image_dict[cell_names[index]][channel]=img
                index+=1
    return image_dict


def to_onehot(my_list,labels):
    return_list=[]
    for i,elem in enumerate(my_list):
        j=np.where(np.unique(labels)==elem)
        return_list.append(np.zeros((len(np.unique(my_list)))))
        return_list[-1][j]=1
    return np.array(return_list)

# Helper functions
def np_image_to_base64(im_matrix):
    im = Image.fromarray(im_matrix)
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url


def interactive_session(u,display_images,colors,namesdf):

    dim=np.shape(u)[1]

    if np.min(display_images)<0:
        display_images=np.array((np.array(display_images)-np.min(display_images)))

    if str(type(display_images[0][0][0]))=="<class 'numpy.float64'>":
        display_images=display_images.astype(np.uint8)


   
    buffer = io.StringIO()

    if dim==3:
        fig = go.Figure(data=[go.Scatter3d(
            x=u[:, 0],
            y=u[:, 1],
            z=u[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=colors,
            )
        )])

    if dim==2:
        fig = go.Figure(data=[go.Scatter(
            x=u[:, 0],
            y=u[:, 1],
            mode='markers',
            marker=dict(
                size=5,
                color=colors,
            )
        )])


    fig.update_traces(
        hoverinfo="none",
        hovertemplate=None,
    )

    fig.update_layout(template='plotly_dark', title="Retinal cell clustering (t-SNE)")
    fig.show()

    #fig.update_layout(
    #    scene=dict(
    #        xaxis=dict(range=[-10,10]),
    #        yaxis=dict(range=[-10,10]), 
    #    )
    #)

    fig.write_html(buffer)

    html_bytes = buffer.getvalue().encode()
    encoded = b64encode(html_bytes).decode()

    app = JupyterDash(__name__)

    app.layout = html.Div(
        className="container",
        children=[
            dcc.Graph(style={'height': '800px'}, id="graph-5", figure=fig, clear_on_unhover=True),
            dcc.Tooltip(id="graph-tooltip-5", direction='bottom'),
            ], style={'width': '80%' , 'display': 'inline-block', 'vertical-align': 'middle'}
    ) 


    @app.callback(
        Output("graph-tooltip-5", "show"),
        Output("graph-tooltip-5", "bbox"),
        Output("graph-tooltip-5", "children"),
        Input("graph-5", "hoverData"),
    )

    def display_hover(hoverData):
        if hoverData is None:
            return False, no_update, no_update


        hover_data = hoverData["points"][0]
        bbox = hover_data["bbox"]
        num = hover_data["pointNumber"]

        im_matrix = display_images[num]
        im_url = np_image_to_base64(im_matrix)
        children = [
            html.Div([
                html.Img(
                    src=im_url,
                    style={"width": "100px", 'display': 'block', 'margin': '0 auto'},
                ),
                html.P("Cell ID: " + str(int(namesdf[num])))#, style={'font-weight': 'bold'})
            ])
        ]

        return True, bbox, children

    if __name__ == "Helper_functions":
    #  app.run_server(mode='inline', debug=True)
        port_num=np.random.randint(100)+8000
        app.run_server(mode='external', debug=True,use_reloader=False,port=port_num)

def make_tile(display_images,u,n,savepath):
    """Makes a tile of images according to a 1-dimensional umap, n is the width of the frame in num of images
    savepath can either be "False' or a path string"""
    assert np.shape(u)[1]==1

    u=u.reshape(len(u))
    sorted_u=np.argsort(u)
    display_ims=np.array(display_images)[sorted_u]
    l=len(u)//n
    for i in range(l):
        for j in range(n):
            if j==0:
                #col_array=overlay(train_data1[i*n],nuctrain_data1[i*n])
                col_array=display_ims[i*n]
            else:
                #col_array=np.hstack((col_array,overlay(train_data1[i*n+j],nuctrain_data1[i*n+j])))
                col_array=np.hstack((col_array,display_ims[i*n+j]))

        if i==0:
            row_array=col_array
        else:
            row_array=np.vstack((row_array,col_array))

    plt.figure(dpi=2000)
    plt.imshow(row_array,cmap="Greys")  
    plt.axis('off')
    if savepath!='False':
        plt.savefig(savepath,bbox_inches="tight")
    plt.show()

            

