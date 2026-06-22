
SAGE-Face (Senior Affective Graph of Emotions)

AffectNet
DFEW
FACES
FERPlus
ElderReact
RafDB


"""
AFFECTNET
Base AffectNet strutturato già Elder e Young

"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\1_Dataset\AffectNet\AFFECTNET_Elder"
Affectnet_Elder/ Anger, Contempt, disgust, fear, happy, neutral, sad, surprise
96x96

DFEW
"C:\Users\ciano\Desktop\Tesi\Dataset\DFEW\Clip\Clip\clip_224x224_16f"
DFEW / Clip / clip_224x224_16f 16 frame per video
"C:\Users\ciano\Desktop\Tesi\Dataset\DFEW\Clip\Clip\clip_224x224_16f\00001\1.jpg"
"C:\Users\ciano\Desktop\Tesi\Dataset\DFEW\Clip\Clip\clip_224x224_16f\00001\16.jpg"
224x224


annotation.xlsx
1happy 2sad  3neutral  4angry 5surprise  6disgust 7fear  order  label
0  6  4  0 0  0 0 1 2
Praticamente l'annotazione è su tutto il video

FACES
"C:\Users\ciano\Desktop\Tesi\Dataset\FACES\004_o_m_a_a.jpg"
"C:\Users\ciano\Desktop\Tesi\Dataset\FACES\168_m_f_s_b.jpg"
2835 x 3453

La seconda lettera: gender
male female

La terza lettera: emotion
anger disgust fear happiness neutral sadness

ELDERREACT
Sono inizialmente clip video, divise ognuna in 16 frame
ElderReact/ElderReact_Frames/dev test train
"C:\Users\ciano\Desktop\Tesi\Dataset\ElderReact\ElderReact_Frames\dev\50_50_35\50_50_35_1.jpg"
"C:\Users\ciano\Desktop\Tesi\Dataset\ElderReact\ElderReact_Frames\dev\50_50_35\50_50_35_16.jpg"

ho a disposizizione i tensori per ogni clip video

Queste le annotazioni per le clip video
"C:\Users\ciano\Desktop\Tesi\Dataset\ElderReact\ElderReact-master\Annotations\dev_labels.txt"
"C:\Users\ciano\Desktop\Tesi\Dataset\ElderReact\ElderReact-master\Annotations\test_labels.txt"
"C:\Users\ciano\Desktop\Tesi\Dataset\ElderReact\ElderReact-master\Annotations\train_labels.txt"

50_50_100.mp4 0 1 1 0 0 1 f 1.6666666666666667
50_50_104.mp4 1 0 0 0 0 0 m 2.3333333333333335
50_50_105.mp4 1 1 0 0 0 1 f 1.6666666666666667
50_50_106.mp4 1 1 0 0 0 1 f 2.0
50_50_22.mp4 0 1 1 0 1 1 f 2.6666666666666665
50_50_24.mp4 0 0 0 1 0 1 f 6.333333333333333

1->filename, 2->Anger, 3->Disgust, 4->Fear, 5->Happiness, 6->Sadness, 7->Surprise, 8->Gender, 9->Valence
224x224


RAF-DB
"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\1_Dataset\Raf-DB\Complete"
Raf-DB\Complete\Anger disgust fear happy neutral sad surprise

"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\1_Dataset\Raf-DB\Complete\Anger\test_0017_aligned.jpg"

100x100
"""
