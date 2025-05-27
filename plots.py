import plotly.express as px
px.imshow(draw_segmentation_map(img, target))


from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(rows=1, cols=2, subplot_titles=("Input", "Ground Truth"))
fig.add_trace(go.Image(z=img.numpy().transpose(1,2,0)*255), 1, 1)
fig.add_trace(go.Image(z=draw_segmentation_map(img, target)*255), 1, 2)





def plot_prediction():
    (img,target) = dataset_test[6]
    imgs = img.unsqueeze(0).to(device) #torch.stack((img,img))
    outs = model2(imgs)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Prediction (all scores)", "Prediction (scores>0.8)"))
    fig.add_trace(go.Image(z=draw_segmentation_map(img, outs[0], score_thres=0.0)*255), 1, 1)
    fig.add_trace(go.Image(z=draw_segmentation_map(img, outs[0], score_thres=0.8)*255), 1, 2)