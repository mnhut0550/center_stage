import tensorflow as tf
import cv2

class face_detector:
    def __init__(self,model_path,model_input_size=(224,224)):
        self.model_path = model_path
        self.size = model_input_size
        self.load_frozen_graph()
        self.pre_inference()
        
    def load_frozen_graph(self):
        with tf.io.gfile.GFile(self.model_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            self.graph = graph_def
            
    def preprocessing_input(self,image):
        self.img = image
        self.shape = self.img.shape[:2]
        img = cv2.resize(self.img,self.size)
        return [img]
    
    def warmup(self,EX_links):
        img = cv2.imread(EX_links)
        img = cv2.resize(img,self.size)
        self.inference(img)
        
    def pre_inference(self):
        self.session = tf.compat.v1.Session()
        tf.import_graph_def(self.graph,name='detector')
        
        self.boxes = self.session.graph.get_tensor_by_name("detection_boxes:0")
        self.scores = self.session.graph.get_tensor_by_name("detection_scores:0")
    
    def inference(self,img,threshold=0.6):
        img = self.preprocessing_input(img)
        feed_input = {"image_tensor:0":img}
        preds = self.session.run([self.boxes,self.scores],
                                 feed_dict=feed_input)
        boxes = preds[0][0]
        scores = preds[1][0]
        
        scores = [s for s in scores if 1.0 >=s >= threshold]
        boxes = boxes[:len(scores)]
        boxes = [self.get_realbox(box) for box in boxes]
        
        return boxes,scores

    def get_realbox(self,box):
        start_x, start_y, end_x, end_y = abs(box[1])*self.shape[1], abs(box[0])*self.shape[0], abs(box[3])*self.shape[1], abs(box[2])*self.shape[0]
        start_x,start_y,end_x,end_y = int(start_x),int(start_y),int(end_x),int(end_y)
        
        return start_x,start_y,end_x,end_y
