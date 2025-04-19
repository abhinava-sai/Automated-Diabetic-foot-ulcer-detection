PROBLEM IDENTIFICATION & OBJECTIVE:


Diabetic Foot Ulcers (DFUs) are a serious concern for people with diabetes, often leading to
infections, gangrene, or even amputations if not treated early. Diagnosing these ulcers typically
relies on a doctor’s experience, which can lead to mistakes or delays in treatment. In rural areas,
accessing specialized care is even harder, making early detection challenging. On top of that, AI
models used for diagnosis often struggle with accuracy due to limited and unbalanced data. This
project aims to tackle these issues by creating an advanced deep learning system that can detect
DFUs more reliably, offering clear explanations for its results and helping doctors make faster,
more confident decisions.

OBJECTIVE:


This project aims to develop a deep learning-based automated DFU detection and classification
system that:

- Utilizes state-of-the-art CNN architectures (EfficientNet, VGG16, ResNet50,
InceptionV3) to enhance accuracy.
- Employs transfer learning and hybrid models to improve feature extraction and
classification.
- Implements Grad-CAM visualizations for interpretability, helping clinicians understand
AI-driven decisions.
- Addresses data imbalance issues through augmentation techniques, ensuring robust model
performance.
- Provides a scalable and deployable solution for real-time DFU detection, improving
patient outcomes and reducing healthcare burden.

EXISTING SYSTEM:


Currently, Diabetic Foot Ulcers (DFUs) are usually diagnosed by doctors through a physical
examination. They rely on their experience and judgment to assess the severity of the ulcers,
sometimes using additional tools like X-rays or MRI scans. While this method can work well, it
often depends heavily on the doctor’s expertise, which can lead to differing opinions and the risk of
human error.
In rural and underserved areas, finding specialists like podiatrists or diabetic care experts can be
difficult. This lack of access often results in delayed diagnoses and treatment. Although some AI
systems exist to assist with DFU detection, they tend to struggle with accuracy due to limited and
unbalanced data. Additionally, these systems often work as "black boxes," providing results without
explaining how they arrived at those conclusions. This makes it harder for doctors to trust and rely
on the AI’s suggestions.knowledge for feature extraction.


Training & Evaluation:


Used early stopping and learning rate reduction techniques to prevent overfitting.
Assessed performance with accuracy, precision, recall, and F1-score metrics.
Enhanced model interpretability with Grad-CAM visualizations to highlight key decision areas. 

PROPOSED SYSTEM METHODOLOGY :


The proposed system is designed to automate the detection and class of diabetic foot ulcers (DFUs)
the usage of deep learning techniques. It aims to cope with demanding situations associated with
records imbalance, interpretability, and generalization in DFU detection with the aid of leveraging
switch gaining knowledge of and ensemble models. the key additives of the machine
encompass:facts Preprocessing and Augmentation:pictures are resized to a uniform shape of
224x224 pixels to make sure compatibility across fashions.facts augmentation techniques which
includes rotation, zoom, horizontal flipping, and shift transformations are carried out to increase
dataset range and enhance model generalization.
Version development: Multiple pre-skilled CNN architectures, along with EfficientNetB0,
VGG16, ResNet50, and InceptionV3, are employed to extract deep hierarchical capabilities. A
hybrid model is advanced via combining EfficientNet, VGG16, and ResNet50 right into a single
structure, merging the feature maps from these models earlier than passing them to a dense class
head.

Training and high-quality-Tuning: 

A two-degree schooling technique is carried out. to start with,
the pre-trained models are frozen, and simplest the category head is skilled. eventually, the whole
version is excellent-tuned with a lower studying rate to improve overall performance on the DFU
dataset.class weights are computed to address the imbalance among healthful and ulcer pix, making
sure balanced getting to know.

Evaluation and Interpretability: 

Model performance is evaluated the usage of accuracy,
precision, don't forget, F1-rating, AUC-ROC, and precision-bear in mind curves.Grad-CAM
(Gradient-weighted magnificence Activation Mapping) is hired to visualize the regions of hobby
within the foot images that encouraged the model’s predictions, improving interpretability and
helping clinical adoption.

Deployment and destiny Scope: 

The skilled models can be packaged into a user-pleasant
interface, enabling real-time DFU detection.destiny improvements could include increasing the
dataset, applying semi-supervised gaining knowledge of, and deploying the version in a cellular
software for accessibility in far flung regions.

SYSTEM ARCHITECTURE:


Data would be collated and integrated over time throughout the various modules of the architecture.
The above features of the system architecture hence integrate all elements to offer model training
and prediction besides monitoring.

Sources: The system will work on collections of data sourced from various sources, like clinical
data files which, in examples, are used to contain patient medical histories just like those in
electronic health records or lifestyle data like activity levels or typical diets.

Data Collection & Preprocessing:


Curated a dataset of healthy and ulcer foot images.Applied normalization, augmentation, and class
balancing to enhance model robustness.

Model Development:


Implemented multiple CNN architectures: EfficientNet, VGG16, ResNet50, InceptionV3, and
custom Hybrid models.Applied transfer learning with pretrained models to leverage existing
knowledge for feature extraction.

Training & Evaluation:


Used early stopping and learning rate reduction techniques to prevent overfitting.
Assessed performance with accuracy, precision, recall, and F1-score metrics.
Enhanced model interpretability with Grad-CAM visualizations to highlight key decision areas. 

IMPLEMENTATION:
This project focuses on building a deep learning-based system to detect and classify Diabetic Foot
Ulcers (DFUs) using medical images. The implementation process includes several practical steps:
1. Data Collection and Preparation:
First, a set of DFU images will be gathered from trusted medical sources. These images will be
cleaned, resized, and standardized to maintain consistency. To make the model more reliable, data
augmentation techniques will be applied, simulating various real-world scenarios by adjusting
lighting, angles, and contrast.
2. Model Selection and Training:
Popular AI models like EfficientNet, VGG16, ResNet50, and InceptionV3 will be tested and
compared to identify the most effective one. Transfer learning, using pre-trained models, will speed
up the process and improve accuracy. The model’s parameters will be fine-tuned to ensure it
performs well.
3. Creating a Hybrid Model:
To get the best results, a hybrid approach will combine the strengths of multiple models. This will
allow the system to better analyze different image features and make more accurate predictions.
4. Making AI Understandable:
Since doctors need to trust the system’s decisions, Grad-CAM (Gradient-weighted Class Activation
Mapping) will be used. This tool highlights the areas in an image that influenced the model’s
decision, providing clear visual explanations.
5. Evaluation and Testing:
The model will be evaluated using accuracy, precision, recall, and F1 scores to measure its
performance. Cross-validation will also be conducted to ensure the model works well with new,
unseen data.

Key Findings:


- VGG16 achieved high accuracy and generalization, making it a reliable model for DFU
detection.
- GoogleNet exhibited perfect classification in this experiment, though further validation on a
larger dataset is recommended.
- Hybrid Model demonstrated strong performance by combining feature extraction
capabilities from multiple architectures.
- EfficientNet struggled with the given dataset, likely due to insufficient fine-tuning or
limited training data.
- Grad-CAM visualizations revealed that the best-performing model is ResNet50 which
focused on the lesion areas, enhancing trust in predictions.

CONCLUSION:


Diabetic Foot Ulcers (DFUs) are a potentially life-threatening complication of diabetes that, if not
detected early, result in serious infections, amputations, and even death. Conventional methods of
diagnosis are heavily dependent on physicians' expertise and physical examinations, hence are
subject to human error, delay, and variability. Having sensed the acute need for a faster, more
accurate, and more reliable diagnostic system, our project harnessed the capability of deep learning
to automate the detection and classification of DFUs.
With extensive experimentation and research, we trained and tested various deep learning models
like EfficientNet, VGG16, ResNet50, InceptionV3, and a Hybrid model. Out of these, VGG16 and
the Hybrid model performed the best, with a maximum of 97% accuracy. What is even more crucial
is that we implemented Grad-CAM visualizations such that the system does not operate like a
"black box" but rather gives transparent, understandable output that physicians can rely upon.
This project is a major milestone towards closing the gap between AI and medicine, providing a
practical and scalable solution that can assist clinicians in making rapid and accurate decisions. Yet,
as with any technology, there is always scope for improvement. Increasing the dataset, further
fine-tuning the models, and implementing the system in real-world clinical environments will make
it more effective and impactful. The long-term objective is to provide early DFU detection to all,
but particularly in rural and disadvantaged areas where specialist services are scarce.
In the long term, incorporating this technology into mobile devices or wearable sensors may
transform the management of diabetes, enabling patients to remotely check their foot condition and
receive prompt medical attention. Through the application of AI, we are not merely creating a
model—we are creating a pathway to a future where healthcare is smarter, faster, and more
patient-focused.

