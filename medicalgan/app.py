import sys
import medicalgan.releases.mnist.mnist_gan as mnist_gan
import medicalgan.releases.fmri_tumour.fmri_tumour_cnn as fmri_tumour_cnn
import medicalgan.releases.adni_alzheimers.adni_alzheimers_cnn as adni_alzheimers_cnn
import medicalgan.releases.oasis.oasis_cnn as oasis_cnn


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
            if task == "train":
                mnist_gan.train()
            # if task == "generate":
                # mvp.generate()
        if release == "fmri_tumour":
            # if task == "train":
                # fmri_tumour_gan.train()
            # if task == "generate":
                # fmri_tumour_gan.generate()
            if task == "detect":
                fmri_tumour_cnn.train()
        # if release == "nih_chest_xray":
            # if task == "train":
                # nih_chest_xray_gan.train()
            # if task == "generate":
                # nih_chest_xray_gan.generate()
            # if task == "detect":
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
            # if task == "train":
                # adni_alzheimers_gan.train()
            # if task == "generate":
                # adni_alzheimers_gan.generate()
            if task == "detect":
                adni_alzheimers_cnn.train(plane)

        if release == "oasis":
            if not len(args) == 3:
                print("\nUnrecognised arguments.\nExpected:\n\t> medicalgan [oasis] [task]\n")
                return
            # if task == "train":
                # adni_alzheimers_gan.train()
            # if task == "generate":
                # adni_alzheimers_gan.generate()
            if task == "detect":
                oasis_cnn.train()
