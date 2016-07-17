function varargout = GUI(varargin)
% GUI MATLAB code for GUI.fig
%      GUI, by itself, creates a new GUI or raises the existing
%      singleton*.
%
%      H = GUI returns the handle to a new GUI or the handle to
%      the existing singleton*.
%
%      GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GUI.M with the given input arguments.
%
%      GUI('Property','Value',...) creates a new GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before GUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to GUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help GUI

% Last Modified by GUIDE v2.5 25-Jun-2016 16:59:56

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @GUI_OpeningFcn, ...
                   'gui_OutputFcn',  @GUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before GUI is made visible.
function GUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to GUI (see VARARGIN)

% Choose default command line output for GUI
handles.output = hObject;

% Add to path
%addpath(genpath(pwd))

% Load data set info 
params = esvm_get_default_params_1;
datasets_info = esvm_get_datasets_info(params.datasets_params);
classes = cellfun(@(x) x.cls_name, datasets_info, 'UniformOutput', false);
handles.datasets_info = datasets_info;
handles.params = params;
handles.classes = classes;
handles.file_ext = params.datasets_params.file_ext;

%initialize feat_name, calibration, hard-negative
handles.feat_name = 'hog';
handles.hard_negative = false;
handles.calibration = false;

% Load model and calibration matrix
[handles.hog_model_hn, handles.hog_model_wo_hn, handles.hog_cal_matrix_hn, handles.hog_cal_matrix_wo_hn,...
 handles.cnn_model_hn, handles.cnn_model_wo_hn, handles.cnn_cal_matrix_hn, handles.cnn_cal_matrix_wo_hn,...
 handles.cnnhog_model_hn, handles.cnnhog_model_wo_hn, handles.cnnhog_cal_matrix_hn, handles.cnnhog_cal_matrix_wo_hn] = esvm_gui_initialize();


format(handles.predicted_img);
format(handles.test_img);
format(handles.predict_1);
format(handles.predict_2);
format(handles.predict_3);
format(handles.predict_4);
format(handles.predict_5);
format(handles.predict_6);
format(handles.predict_7);
format(handles.predict_8);
format(handles.predict_9);
format(handles.predict_10);
format(handles.predict_11);
format(handles.predict_12);
format(handles.predict_13);
format(handles.predict_14);
format(handles.predict_15);
format(handles.predict_16);


%load cnn model
cnn_params = params.features_params.cnn_params;
mcnpath = fullfile('.', 'lib','matconvnet-1.0-beta18','matlab');
run(fullfile(mcnpath,'vl_setupnn.m'));
cnn_path = fullfile('.',cnn_params.model_folder,cnn_params.model_name);
convnet = load(cnn_path);   
handles.convnet = convnet;

% Update handles structure
guidata(hObject, handles);

function format(axes_name)
set(axes_name, 'Color',[0.94 0.94 0.94]);
set(axes_name,'xtick',[]);
set(axes_name,'ytick',[]);
set(axes_name,'visible','off');
set(axes_name,'Units','pixels');


% UIWAIT makes GUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = GUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in feature.
function feature_Callback(hObject, eventdata, handles)
% hObject    handle to feature (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
feat_value = get(hObject,'Value');
switch feat_value
    case 1
        feat_name = 'hog';
    case 2
        feat_name = 'cnn';
    case 3
        feat_name = 'cnnhog';
end
handles.feat_name = feat_name;
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function feature_CreateFcn(hObject, eventdata, handles)
% hObject    handle to feature (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
     set(hObject,'BackgroundColor','white');
end

set(hObject, 'String', {'HoG', 'CNN', 'HoG-CNN'});

% --- Executes on button press in checkbox1.
function checkbox1_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox1
value = get(hObject,'Value');
switch value
    case 0
        calibration = false;
    case 1
        calibration = true;
end
handles.calibration = calibration;
guidata(hObject, handles);

% --- Executes on button press in checkbox2.
function checkbox2_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox2
value = get(hObject,'Value');
switch value
    case 0
        hn = false;
    case 1
        hn = true;
end
handles.hard_negative = hn;
guidata(hObject, handles);


% --- Executes on button press in checkbox3.
function checkbox3_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox3


% --------------------------------------------------------------------
function open_img_ClickedCallback(hObject, eventdata, handles)
% hObject    handle to open_img (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in img_selector.
function handels = img_selector_Callback(hObject, eventdata, handles)
% hObject    handle to img_selector (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
file_ext = sprintf('*.%s',handles.file_ext);
[FileName,PathName,FilterIndex] = uigetfile({file_ext}, 'Select a test image');

parts = strsplit(PathName, '/');
original_class = parts{end-2};
%fprintf(1, original_class);
if any(ismember(handles.classes,original_class))
    set(handles.original_cls,'String', original_class)
else
    set(handles.original_cls,'String', 'Unknown')
end
handles.test_img_filer = fullfile(PathName,FileName);

img = imread(fullfile(PathName,FileName));

handles.test_img_height = size(img,1);
fprintf(num2str(size(img,1)));
handles.test_img_width = size(img,2);
axes(handles.test_img);
imshow(img);
% Update handles structure
guidata(hObject, handles);


% --- Executes on button press in test_now.
function test_now_Callback(hObject, eventdata, handles)
% hObject    handle to test_now (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

%calculate the size of samll window for showing predictions
position = get(handles.predict_1,'Position');
width = position(3) - position(1);
height = position(4) - position(2);

%get prediction results
test_img = handles.test_img_filer
feat_name = handles.feat_name;
hard_negative = handles.hard_negative;
calibration = handles.calibration;
predictions = esvm_gui_predict(test_img, feat_name, hard_negative, calibration, handles);
% show the predictions
for i=1:length(predictions)
    for j = 1:length(predictions{i})
        which = (i-1) * 4 + j;
        prediction = predictions{i}{j};
        img = imread(prediction.img_filer);
        if i == 1
            fprintf('%s \n',prediction.img_filer);
        end
        img = imresize(img, [height width]);
        tag_score = sprintf('score_%d',which);
        tag_predict = sprintf('predict_%d',which);
        axes(getfield(handles,tag_predict));
        %setfield(getfield(handles,tag_predict),'img_filer', prediction.img_filer);
        imshow(img);
        score_string = sprintf('s=%f', prediction.score);
        set(getfield(handles,tag_score),'String',score_string);
    end
end

set(handles.predicted_cls,'String', handles.classes(predictions{1}{1}.cls_idx));
set(handles.panel_similar_1,'Title', sprintf('Most similar: %s', handles.classes{predictions{1}{1}.cls_idx}));
set(handles.panel_similar_2,'Title', sprintf('2nd most similar: %s', handles.classes{predictions{2}{1}.cls_idx}));              
set(handles.panel_similar_3,'Title', sprintf('3rd most similar: %s', handles.classes{predictions{3}{1}.cls_idx})); 
set(handles.panel_similar_4,'Title', sprintf('4th most similar: %s', handles.classes{predictions{4}{1}.cls_idx}));


show_predicted_img(predictions{1}{1}.img_filer, handles);
handles.current_show_img_filer = predictions{1}{1}.img_filer;
handles.predictions = predictions;
predictions{1}{1}
guidata(hObject, handles);



% --- Executes on button press in try_my_luck.
function try_my_luck_Callback(hObject, eventdata, handles)
% hObject    handle to try_my_luck (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.test_img_filer = generate_random_test_img_filer(handles.datasets_info);
handles.test_img_filer
%display test image

parts = strsplit(handles.test_img_filer, '/');
original_class = parts{end-2};
%fprintf(1, original_class);
if any(ismember(handles.classes,original_class))
    set(handles.original_cls,'String', original_class)
else
    set(handles.original_cls,'String', 'Unknown')
end

img = imread(handles.test_img_filer);

handles.test_img_height = size(img,1);
fprintf(num2str(size(img,1)));
handles.test_img_width = size(img,2);
axes(handles.test_img);
imshow(img);

%calculate the size of samll window for showing predictions
position = get(handles.predict_1,'Position');
width = position(3) - position(1);
height = position(4) - position(2);

%get prediction results
test_img = handles.test_img_filer;
feat_name = handles.feat_name;
hard_negative = handles.hard_negative;
calibration = handles.calibration;
predictions = esvm_gui_predict(test_img, feat_name, hard_negative, calibration, handles);
% show the predictions
for i=1:length(predictions)
    for j = 1:length(predictions{i})
        which = (i-1) * 4 + j;
        prediction = predictions{i}{j};
        img = imread(prediction.img_filer);
        img = imresize(img, [height width]);
        tag_score = sprintf('score_%d',which);
        tag_predict = sprintf('predict_%d',which);
        axes(getfield(handles,tag_predict));
        %setfield(getfield(handles,tag_predict),'img_filer', prediction.img_filer);
        imshow(img);
        %set(getfield(handles,tag_predict),'ButtonDownFcn',@dosomething); 
        score_string = sprintf('s=%f', prediction.score);
        set(getfield(handles,tag_score),'String',score_string);
    end
end

set(handles.predicted_cls,'String', handles.classes(predictions{1}{1}.cls_idx));
set(handles.panel_similar_1,'Title', sprintf('Most similar: %s', handles.classes{predictions{1}{1}.cls_idx}));
set(handles.panel_similar_2,'Title', sprintf('2nd most similar: %s', handles.classes{predictions{2}{1}.cls_idx}));              
set(handles.panel_similar_3,'Title', sprintf('3rd most similar: %s', handles.classes{predictions{3}{1}.cls_idx})); 
set(handles.panel_similar_4,'Title', sprintf('4th most similar: %s', handles.classes{predictions{4}{1}.cls_idx}));

show_predicted_img(predictions{1}{1}.img_filer, handles);
handles.current_show_img_filer = predictions{1}{1}.img_filer;
handles.predictions = predictions;
predictions{1}{1}
guidata(hObject, handles);



% --- Executes on key press with focus on img_selector and none of its controls.
function img_selector_KeyPressFcn(hObject, eventdata, handles)
% hObject    handle to img_selector (see GCBO)
% eventdata  structure with the following fields (see MATLAB.UI.CONTROL.UICONTROL)
%	Key: name of the key that was pressed, in lower case
%	Character: character interpretation of the key(s) that was pressed
%	Modifier: name(s) of the modifier key(s) (i.e., control, shift) pressed
% handles    structure with handles and user data (see GUIDATA)


% --- Executes during object creation, after setting all properties.
function predicted_img_CreateFcn(hObject, eventdata, handles)
% hObject    handle to predicted_img (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate predicted_img


% --- Executes during object creation, after setting all properties.
function test_img_CreateFcn(hObject, eventdata, handles)
% hObject    handle to test_img (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate test_img

% --- Executes on mouse press over axes background.
function predict_2_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to predict_2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
parts = strsplit(get(hObject,'Tag'),'_');
which = parts(2);
i = which / 4;
j = mod(which, 4);
img_filer = handles.predictions{i}{j}.img_filer;
%img_filer = get(hObject,'img_filer')
show_predicted_img(img_filer, handles);
handles.current_show_img_filer = predictions{1}{1}.img_filer;
guidata(hObject, handles);


function show_predicted_img(img_filer, handles)
img = imread(img_filer);
img = imresize(img, [handles.test_img_height handles.test_img_width]);
axes(handles.predicted_img);
imshow(img);


% --------------------------------------------------------------------
function dosomething
fprintf('ddddd');
