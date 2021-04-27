import sys
import medicalgan.releases.mnist.mnist_gan as mnist_gan
import medicalgan.releases.fmri_tumour.fmri_tumour_cnn as fmri_tumour_cnn
import medicalgan.releases.adni_alzheimers.adni_alzheimers_cnn as adni_alzheimers_cnn
import medicalgan.releases.oasis.oasis_cnn as oasis_cnn
import medicalgan.releases.oasis.oasis_gan as oasis_gan
import medicalgan.releases.oasis.oasis_wgan as oasis_wgan
import medicalgan.releases.oasis.oasis_crossval_cnn as oasis_crossval_cnn
import medicalgan.releases.oasis.oasis_transferlearning as oasis_transferlearning_cnn


def run():
    """
    Generation of medical images using Generative Adversarial Networks.
    """
    args = sys.argv
    if not len(args) > 2:
        print("\nPlease specify data and task:\
               \n>  medicalgan [data] [task]\n")
        return
    else:
        release, task = args[1], args[2]
        if release == "mnist":
            if task == "train_gan":
                mnist_gan.train()
            # if task == "generate":
                # mvp.generate()
        if release == "fmri_tumour":
            # if task == "train":
                # fmri_tumour_gan.train()
            # if task == "generate":
                # fmri_tumour_gan.generate()
            if task == "train_cnn":
                fmri_tumour_cnn.train()
        # if release == "nih_chest_xray":
            # if task == "train":
                # nih_chest_xray_gan.train()
            # if task == "generate":
                # nih_chest_xray_gan.generate()
            # if task == "train_cnn":
            #     nih_chest_xray_cnn.train()
        if release == "adni":
            if not len(args) == 4:
                print("\nPlease specify MRI plane (i.e. transverse, sagital or coronal):\
                       \n>  medicalgan [adni] [task] [plane]\n")
                return
            plane = args[3]
            if not plane in ["transverse", "sagital", "coronal"]:
                print("\nUnsupported plane, use transverse or sagital\n")
                return
            # if task == "train_gan":
                # adni_alzheimers_gan.train()
            # if task == "generate":
                # adni_alzheimers_gan.generate()
            if task == "train_cnn":
                adni_alzheimers_cnn.train(plane)

        if release == "oasis":
            if not (len(args) == 5 or len(args) == 6):
                print("\nPlease specify arguments:\
                       \n>  medicalgan [oasis] [task] [plane] [depth]\n")
                return
            plane = args[3]
            if not plane in ["transverse", "sagital", "coronal"]:
                print("\nUnsupported plane, use transverse or sagital\n")
                return
            depth = args[4]
            if not depth in ["single", "multi"]:
                print("\nUnsupported depth, use single or multi\n")
                return
            if task == "train_gan":
                label = args[5]
                if not label in ["0", "1"]:
                    print("\nPlease specify class to generate (i.e. 0 or 1):\
                       \n>  medicalgan [oasis] [task] [plane] [depth] [label]\n")
                    return
                label = int(label)
                oasis_gan.train(plane, depth, label)
            if task == "train_wgan":
                label = args[5]
                if not label in ["0", "1"]:
                    print("\nPlease specify class to generate (i.e. 0 or 1):\
                       \n>  medicalgan [oasis] [task] [plane] [depth] [label]\n")
                    return
                label = int(label)
                oasis_wgan.train(plane, depth, label)
            # if task == "generate":
                # oasis_gan.generate()
            if task == "train_cnn":
                oasis_cnn.train(plane, depth)
            if task == "train_transferlearning_cnn":
                oasis_transferlearning_cnn.train(plane, depth)
            if task == "train_crossval_cnn":
                oasis_crossval_cnn.train(plane, depth)
            if task == "evaluate_cnn":
                oasis_cnn.evaluate(plane, depth)
