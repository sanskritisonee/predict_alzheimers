"""
Here we do inference on a DICOM volume, constructing the volume first, and then sending it to the
clinical archive

This code will do the following:
    1. Identify the series to run HippoCrop.AI algorithm on from a folder containing multiple studies
    2. Construct a NumPy volume from a set of DICOM files
    3. Run inference on the constructed volume
    4. Create report from the inference
    5. Call a shell script to push report to the storage archive
"""

import os
import datetime
import time
import shutil
import subprocess
import argparse
import json
from pathlib import Path

import numpy as np
import pydicom

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from inference.UNetInferenceAgent import UNetInferenceAgent


def _load_report_font(size):
    """Load a bundled font when available, otherwise fall back to PIL's default."""
    font_path = Path(__file__).resolve().parent / "assets" / "Roboto-Regular.ttf"
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size=size)

    print(f"Warning: font asset not found at {font_path}. Falling back to default font.")
    return ImageFont.load_default()


def _header_value(header, field, default="N/A"):
    """Read a DICOM field safely so report generation does not fail on missing metadata."""
    value = getattr(header, field, default)
    return str(value) if value not in (None, "") else default


def _discover_study_dir(routing_dir):
    """Accept either a routed-studies folder or a single study folder."""
    subdirs = [x for x in routing_dir.iterdir() if x.is_dir()]
    if subdirs:
        return sorted(subdirs, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return routing_dir


def load_runtime_config(config_path):
    """Load PACS/DICOM runtime settings from a JSON file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in config file: {path}")
    return data


def resolve_arg(args, config, key, default=None):
    """Prefer CLI value, then config file value, then fallback default."""
    value = getattr(args, key)
    if value is not None:
        return value
    return config.get(key, default)

def load_dicom_volume_as_numpy_from_list(dcmlist):
    """Loads a list of PyDicom objects a Numpy array.
    Assumes that only one series is in the array

    Arguments:
        dcmlist {list of PyDicom objects} -- path to directory

    Returns:
        tuple of (3D volume, header of the 1st image)
    """

    # In the real world you would do a lot of validation here
    slices = [np.flip(dcm.pixel_array).T for dcm in sorted(dcmlist, key=lambda dcm: dcm.InstanceNumber)]

    # Make sure that you have correctly constructed the volume from your axial slices!
    hdr = dcmlist[0]

    # We return header so that we can inspect metadata properly.
    # Since for our purposes we are interested in "Series" header, we grab header of the
    # first file (assuming that any instance-specific values will be ighored - common approach)
    # We also zero-out Pixel Data since the users of this function are only interested in metadata
    hdr.PixelData = None
    return (np.stack(slices, 2), hdr)

def get_predicted_volumes(pred):
    """Gets volumes of two hippocampal structures from the predicted array

    Arguments:
        pred {Numpy array} -- array with labels. Assuming 0 is bg, 1 is anterior, 2 is posterior

    Returns:
        A dictionary with respective volumes
    """

    # TASK: Compute the volume of your hippocampal prediction
    volume_ant = np.sum(pred == 1)
    volume_post = np.sum(pred == 2)
    total_volume = volume_ant + volume_post

    return {"anterior": volume_ant, "posterior": volume_post, "total": total_volume}

def create_report(inference, header, orig_vol, pred_vol):
    """Generates an image with inference report

    Arguments:
        inference {Dictionary} -- dict containing anterior, posterior and full volume values
        header {PyDicom Dataset} -- DICOM header
        orig_vol {Numpy array} -- original volume
        pred_vol {Numpy array} -- predicted label

    Returns:
        PIL image
    """

    # The code below uses PIL image library to compose an RGB image that will go into the report
    # A standard way of storing measurement data in DICOM archives is creating such report and
    # sending them on as Secondary Capture IODs (http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_A.8.html)
    # Essentially, our report is just a standard RGB image, with some metadata, packed into
    # DICOM format.

    pimg = Image.new("RGB", (1000, 1000))
    draw = ImageDraw.Draw(pimg)

    header_font = _load_report_font(size=40)
    main_font = _load_report_font(size=20)

    slice_nums = [orig_vol.shape[2]//3, orig_vol.shape[2]//2, orig_vol.shape[2]*3//4] # is there a better choice?

    # Get volumes
    anterior_vol = inference["anterior"]
    posterior_vol = inference["posterior"]
    total_vol = inference["total"]

    # Create the report here and show information that we think would be relevant to
    # clinicians.
    draw.text((350, 20), "HippoVolume.AI", (255, 255, 255), font=header_font)
    draw.multiline_text((10, 120),
                         f"Patient ID: {_header_value(header, 'PatientID')}\n"
                        f"Patient Name: {_header_value(header, 'PatientName')}\n"
                        f"Study Date: {_header_value(header, 'StudyDate')}\n"
                        f"Series Date: {_header_value(header, 'SeriesDate')}\n"
                        f"Series Description: {_header_value(header, 'SeriesDescription')}\n"
                        f"Study Description : {_header_value(header, 'StudyDescription')}\n"
                        f"Modality: {_header_value(header, 'Modality')}\n"
                        f"Image Type: {_header_value(header, 'ImageType')}\n"
                        f"Anterior volume: {anterior_vol}\n"
                        f"Posterior volume: {posterior_vol}\n"
                        f"Total volume: {total_vol} \n"
                        f"LEGEND: Predicted Anterior volume - Grey, Predicted Posterior volume - White\n",
                         (255, 255, 255), font=main_font)

    ## Slice 1
    slice = orig_vol[slice_nums[0], :, :]
    nd_img = np.flip((slice/np.max(slice))*0xff).T.astype(np.uint8)
    # This is how you create a PIL image from numpy array
    pil_i = Image.fromarray(nd_img, mode="L").convert("RGBA").resize((200,200))
    # Paste the PIL image into our main report image object (pimg)
    pimg.paste(pil_i, box=(10, 420))

    pred_slice = pred_vol[slice_nums[0], :, :]
    pred_nd_img = np.flip((pred_slice/np.max(pred_vol))*0xff).T.astype(np.uint8)
    pil_i = Image.fromarray(pred_nd_img, mode="L").convert("RGBA").resize((200,200))
    pimg.paste(pil_i, box=(220, 420))

    ## Slice 2
    slice = orig_vol[slice_nums[1], :, :]
    nd_img = np.flip((slice/np.max(slice))*0xff).T.astype(np.uint8)
    # This is how you create a PIL image from numpy array
    pil_i = Image.fromarray(nd_img, mode="L").convert("RGBA").resize((200,200))
    # Paste the PIL image into our main report image object (pimg)
    pimg.paste(pil_i, box=(480, 420))

    pred_slice = pred_vol[slice_nums[1], :, :]
    pred_nd_img = np.flip((pred_slice/np.max(pred_vol))*0xff).T.astype(np.uint8)
    pil_i = Image.fromarray(pred_nd_img, mode="L").convert("RGBA").resize((200,200))
    pimg.paste(pil_i, box=(680, 420))

    return pimg

def save_report_as_dcm(header, report, path):
    """Writes the supplied image as a DICOM Secondary Capture file

    Arguments:
        header {PyDicom Dataset} -- original DICOM file header
        report {PIL image} -- image representing the report
        path {Where to save the report}

    Returns:
        N/A
    """

    # Code below creates a DICOM Secondary Capture instance that will be correctly
    # interpreted by most imaging viewers including our OHIF

    # Set up DICOM metadata fields. Most of them will be the same as original file header
    out = pydicom.Dataset(header)

    out.file_meta = pydicom.Dataset()
    out.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    out.is_little_endian = True
    out.is_implicit_VR = False

    # We need to change class to Secondary Capture
    out.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    out.file_meta.MediaStorageSOPClassUID = out.SOPClassUID

    # Our report is a separate image series of one image
    out.SeriesInstanceUID = pydicom.uid.generate_uid()
    out.SOPInstanceUID = pydicom.uid.generate_uid()
    out.file_meta.MediaStorageSOPInstanceUID = out.SOPInstanceUID
    out.Modality = "OT" # Other
    out.SeriesDescription = "HippoVolume.AI"

    out.Rows = report.height
    out.Columns = report.width

    out.ImageType = r"DERIVED\PRIMARY\AXIAL" # We are deriving this image from patient data
    out.SamplesPerPixel = 3 # we are building an RGB image.
    out.PhotometricInterpretation = "RGB"
    out.PlanarConfiguration = 0 # means that bytes encode pixels as R1G1B1R2G2B2... as opposed to R1R2R3...G1G2G3...
    out.BitsAllocated = 8 # we are using 8 bits/pixel
    out.BitsStored = 8
    out.HighBit = 7
    out.PixelRepresentation = 0

    # Set time and date
    dt = datetime.date.today().strftime("%Y%m%d")
    tm = datetime.datetime.now().strftime("%H%M%S")
    out.StudyDate = dt
    out.StudyTime = tm
    out.SeriesDate = dt
    out.SeriesTime = tm

    out.ImagesInAcquisition = 1

    # We empty these since most viewers will then default to auto W/L
    out.WindowCenter = ""
    out.WindowWidth = ""

    # Data imprinted directly into image pixels is called "burned in annotation"
    out.BurnedInAnnotation = "YES"

    out.PixelData = report.tobytes()

    pydicom.filewriter.dcmwrite(path, out, write_like_original=False)

def get_series_for_inference(path, series_description="HippoCrop"):
    """Reads multiple series from one folder and picks the one
    to run inference on.

    Arguments:
        path {string} -- location of the DICOM files

    Returns:
        Numpy array representing the series
    """

    # Here we are assuming that path is a directory that contains a full study as a collection
    # of files
    # We are reading all files into a list of PyDicom objects so that we can filter them later
    #dicoms = [pydicom.dcmread(os.path.join(path, f)) for f in os.listdir(path)]

    # Create a series_for_inference variable that will contain a list of only
    # those PyDicom objects that represent files that belong to the series that you
    # will run inference on.
    # It is important to note that radiological modalities most often operate in terms
    # of studies, and it will most likely be on you to establish criteria for figuring
    # out which one of the multiple series sent by the scanner is the one you need to feed to
    # your algorithm. In our case it's rather easy - we have reached an agreement with
    # people who configured the HippoCrop tool and they label the output of their tool in a
    # certain way.

    # Initialise array for series
    dicoms = []
    for root, _, files in os.walk(path):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                dicoms.append(pydicom.dcmread(file_path))
            except Exception:
                # Ignore non-DICOM files in routed study folders.
                continue

    series_for_inference = [
        dcm for dcm in dicoms
        if getattr(dcm, "SeriesDescription", "") == series_description
    ]

    if len(series_for_inference) == 0:
        raise ValueError(
            f"Could not find any DICOM series with SeriesDescription='{series_description}' in {path}"
        )

    # Check if there are more than one series (using set comprehension).
    if len({f.SeriesInstanceUID for f in series_for_inference}) != 1:
        raise ValueError(
            f"Found multiple candidate series with SeriesDescription='{series_description}'. "
            "Please narrow the input study folder or change --series-description."
        )

    return series_for_inference

def send_report_to_pacs(report_path, host, port, ae_title, storescu_bin="storescu"):
    """Send the generated report to a PACS/Orthanc endpoint using storescu."""
    cmd = [
        storescu_bin,
        host,
        str(port),
        "-v",
        "-aec",
        ae_title,
        "+r",
        "+sd",
        str(report_path),
    ]

    print("> " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"'{storescu_bin}' was not found on PATH. Install DCMTK or pass --skip-send."
        ) from exc


def parse_args():
    parser = argparse.ArgumentParser(description="Run DICOM inference and build report")
    parser.add_argument("routing_dir", nargs="?", type=str,
                        help="Directory that contains routed studies")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional JSON config file for PACS/DICOM deployment settings")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to trained model.pth (default: model/out/final_model/model.pth)")
    parser.add_argument("--report-path", type=str, default=None,
                        help="Path where report DICOM will be saved")
    parser.add_argument("--series-description", type=str, default=None,
                        help="SeriesDescription value used to select the MRI series for inference")
    parser.add_argument("--skip-send", action="store_true",
                        help="Create the report DICOM but do not send it to Orthanc via storescu")
    parser.add_argument("--send-host", type=str, default=None,
                        help="PACS/Orthanc hostname for storescu")
    parser.add_argument("--send-port", type=int, default=None,
                        help="PACS/Orthanc port for storescu")
    parser.add_argument("--send-aet", type=str, default=None,
                        help="Called AE title for storescu")
    parser.add_argument("--storescu-bin", type=str, default=None,
                        help="Path or command name for the storescu executable")
    parser.add_argument("--cleanup-study", action="store_true",
                        help="Delete the processed study folder after a successful run")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = load_runtime_config(args.config) if args.config else {}

    repo_root = Path(__file__).resolve().parents[1]
    routing_dir_value = resolve_arg(args, config, "routing_dir")
    if routing_dir_value is None:
        raise ValueError("routing_dir is required. Pass it on the command line or via --config.")
    routing_dir = Path(routing_dir_value)
    default_model = repo_root / "out" / "final_model" / "model.pth"
    default_report = repo_root / "out" / "reports" / "report.dcm"
    model_path_value = resolve_arg(args, config, "model_path", str(default_model))
    report_path_value = resolve_arg(args, config, "report_path", str(default_report))
    series_description = resolve_arg(args, config, "series_description", "HippoCrop")
    send_host = resolve_arg(args, config, "send_host", "127.0.0.1")
    send_port = int(resolve_arg(args, config, "send_port", 4242))
    send_aet = resolve_arg(args, config, "send_aet", "HIPPOAI")
    storescu_bin = resolve_arg(args, config, "storescu_bin", "storescu")
    cleanup_study = bool(config.get("cleanup_study", False)) or args.cleanup_study
    skip_send = bool(config.get("skip_send", False)) or args.skip_send

    model_path = Path(model_path_value)
    report_save_path = Path(report_path_value)
    report_save_path.parent.mkdir(parents=True, exist_ok=True)

    if not routing_dir.exists():
        raise FileNotFoundError(f"Routing directory not found: {routing_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    study_dir = _discover_study_dir(routing_dir)

    print(f"Looking for series to run inference on in directory {study_dir}...")

    series = get_series_for_inference(study_dir, series_description=series_description)
    volume, header = load_dicom_volume_as_numpy_from_list(series)
    print(f"Found series of {volume.shape[2]} axial slices")

    print("HippoVolume.AI: Running inference...")
    # Use the UNetInferenceAgent class and model parameter file from the previous section
    inference_agent = UNetInferenceAgent(
        device="cpu",
        parameter_file_path=str(model_path))

    # Run inference
    # single_volume_inference_unpadded takes a volume of arbitrary size
    # and reshapes y and z dimensions to the patch size used by the model before
    # running inference.
    pred_label = inference_agent.single_volume_inference_unpadded(np.array(volume))
    pred_volumes = get_predicted_volumes(pred_label)

    # Create and save the report
    print("Creating and pushing report...")
    report_img = create_report(pred_volumes, header, volume, pred_label)
    save_report_as_dcm(header, report_img, str(report_save_path))

    # Send report to our storage archive
    # Write a command line string that will issue a DICOM C-STORE request to send our report
    # to our Orthanc server (that runs on port 4242 of the local machine), using storescu tool
    if skip_send:
        print(f"Report saved locally at {report_save_path}. Skipping storescu send.")
    else:
        send_report_to_pacs(
            report_path=report_save_path,
            host=send_host,
            port=send_port,
            ae_title=send_aet,
            storescu_bin=storescu_bin,
        )

    if cleanup_study:
        # Sleep to let downstream systems finish reading before cleanup.
        time.sleep(2)
        shutil.rmtree(study_dir, onerror=lambda f, p, e: print(f"Error deleting: {e[1]}"))
        print(f"Removed processed study directory: {study_dir}")

    print(f"Inference successful on {header['SOPInstanceUID'].value}, out: {pred_label.shape}",
          f"volume ant: {pred_volumes['anterior']}, ",
          f"volume post: {pred_volumes['posterior']}, total volume: {pred_volumes['total']}")
