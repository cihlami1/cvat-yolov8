# Training YOLO models on custom data and adding new labels

## 1. Extract images for ROS node (rosbags)
* [Extraction guide](http://wiki.ros.org/rosbag/Tutorials/Exporting%20image%20and%20video%20data)

## 2. Label images
We want to avoid using [Roboflow](https://roboflow.com/) because the datasets are public when using the free plan. The best option seems to be [CVAT](https://www.cvat.ai/), which can run on a local computer and also supports automatic annotation using custom models.

### Setup
1. **[DONE] Install CVAT on your computer**
   * We are planning to deploy it on a remote server with HTTPS access, so this step will be unnecessary in the future
   * [Installation guide](https://opencv.github.io/cvat/docs/administration/basics/installation/) show steps for Ubuntu 18.04 LTS but the same steps can be used on Ubuntu 20.04 LTS
   * We didn't try the installation steps for Windows and Mac OS
   * You can mount a shared folder to the docker containers to easily access your images later:
     * You need to stop all the running CVAT containers with `docker compose down`
     * Follow the [guide](https://opencv.github.io/cvat/docs/administration/basics/installation/#share-path)
     * If you want to load images from a shared folder and use automatic annotation, you need to mount the shared folder also to `cvat_worker_annotation` in `docker-compose.override.yml`:
        ```yaml
        cvat_worker_annotation:
            ...
            volumes:
                ...
                - cvat_share:/home/django/share:ro
        ```
     * Then you can start the containers using `docker compose up`
2. **[DONE] Optional: Add automatic annotation**
   * This step is for easier and faster labeling
   * Follow the [installation guide](https://opencv.github.io/cvat/docs/administration/advanced/installation_automatic_annotation/)
   * Select a model from already prepared [YOLO models](https://github.com/opencv/cvat#deep-learning-serverless-functions-for-automatic-labeling) and deploy it with CVAT scripts
        ```bash
        ./deploy_cpu.sh <path-to-the-model>
        ./deploy_gpu.sh <path-to-the-model>
        ```
   * You can also deploy your custom model, but the steps are more complicated: [guide](https://opencv.github.io/cvat/docs/manual/advanced/serverless-tutorial/#adding-your-own-dl-models)

3. **Images annotation**
   * Check if the docker containers are running and connect to the server using a web browser (official support is just for Chrome, but Firefox works fine) 
   * Create a new project (if it doesn't already exist) and set all the labels that you need as bounding boxes - select labels from pre-trained YOLO models that you want to keep and add a new one if needed (YOLO was trained on the [COCO dataset](https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/))
   * Create a [new task](https://opencv.github.io/cvat/docs/manual/basics/create_an_annotation_task/), where you import your extracted images, and assign it to the previously created project - this will assign your labels to the task
   * If you mounted a shared folder with your images, you can select them from there for faster processing
   * If you want to use automatic annotation on the whole dataset, you need to go to [task details](https://opencv.github.io/cvat/docs/manual/basics/task-details/), open `Action` menu and select `Automatic annotation` (it can not be done from the [tasks page](https://opencv.github.io/cvat/docs/manual/basics/tasks-page/) for an unknown reason). Then you can select the model you want to use.
   * For manual annotation, open the automatically created job and create your bounding boxes: [options](https://opencv.github.io/cvat/docs/getting_started/#annotation) - you can also use automatic annotation on a single image
   * When you are done, you can [export](https://opencv.github.io/cvat/docs/getting_started/#export-dataset) your dataset - select the YOLO format

## 3. Learn a pre-trained YOLO model on the custom dataset
   * If you created your dataset using CVAT, you need to additionally create [dataset.yaml file](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#11-create-datasetyaml)
   * Check if you have a good [directories organization](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#13-organize-directories)
   * Select YOLO version - we recommend using [YOLOv8](https://github.com/ultralytics/ultralytics)
   * Create Python program to train the pre-trained model on your custom dataset and save the model: [example](https://github.com/ultralytics/ultralytics#python)

> **&#9432; NOTE:** At first you can annotate smaller number of images, i.e. 500 images, with even distribution of all labels, **including the new ones**, and train the model on this dataset. It is not enough to get perfect model, but you can use it to predict the rest of your images and then just correct or add the labels - it will speed up the annotation process.

## 4. Add custom YOLO v8 based model to CVAT
  * We prepared files for YOLO v8 deployment to CVAT in `deploy_yolov8/`, and based on them, you can create your custom model and add it to the annotator
  * First thing you need to do is to create `funcion.yaml` and `function-gpu.yaml` (for GPU support) files
  * The description of the parameters can be found in [docs](https://opencv.github.io/cvat/docs/manual/advanced/serverless-tutorial/#dl-model-as-a-serverless-function)
  * Make sure you renamed the model's name and also name of the docker image and container
    ```yaml
    metadata:
        name: CHANGE NAME  # part of the docker container name
        ...
        annotation:
            name: CHANGE NAME # a display name 
            ...
            spec: ADD/REMOVE LABELS based on your model # the list of labels which the model supports
    
    spec:
        description: CHANGE DESCRIPTION
        ...
        build:
            image: CHANGE NAME # the name of your docker image
            ...
    ```

  * Then add your saved model in `.pt` format to the same folder as the function.yaml file
    * Change the name of model that should be loaded in `main.py` on line 13
      ```python
      13:   model = YOLO("/opt/nuclio/<name-of-your-model>.pt")
      ```
      
  * Now we can deploy your model on our server - if you send it to us (we are not prophets)

## 5. ROS 1 node for online detection
We created ROS 1 node that can run online detection with YOLOv8 based models on data from `sensor_msgs/Image` message
* Tested for ROS Noetic on Ubuntu 20.04 LTS.
* The node subscribes to topic `/sensor_stack/cameras/stereo_front/zed_node/left_raw/image_raw_color`. You can change that by adding following line to the launch file.
  ```xml
  <remap from="/sensor_stack/cameras/stereo_front/zed_node/left_raw/image_raw_color" to="<your-topic>"/>
  ```
* It publishes to topic named `/detector`
* To deploy your custom model:
  1. Create config `.yaml` file with definition of all possible labels (see prepared `.yaml` files in `detector/config/` for reference) 
  2. Put your model to the `detector/model/` directory 
  3. Then create `.launch` file in `detector/launch/` directory (you can copy e.g `yolov8.launch`) and change the value of `yolo_model` and `model_config` accordingly
  4. Run `roslaunch detector <your-launch-file>`