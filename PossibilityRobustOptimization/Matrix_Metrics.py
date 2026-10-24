# -*- coding: utf-8 -*-
"""
Read RayStation-exported NumPy fraction doses or sum DICOM RT Dose files.
DICOM inputs additionally support DVH metrics for a selected RT Structure Set ROI.

NumPy arrays are converted from the exported numeric values to Gy using
NPY_VALUE_SCALE and summed on matching array shapes. The axis order is retained
exactly as saved. Because .npy files contain no patient-coordinate geometry,
NumPy inputs support dose validation and summation but not ROI-based DVH metrics.

For DICOM inputs, fraction doses are converted to Gy with DoseGridScaling and
summed voxel by voxel on their common dose grid. Dx and Vx follow RayStation
11B+ discrete statistics without DVH interpolation. ROI relative-volume
reconstruction is an independent approximation of RayStation's proprietary
shape-based method. Undefined metrics, such as D0.03cc for a smaller ROI,
are reported as NaN.

Configure the paths and structure name below, then run this file in Cursor.
Relative paths are resolved from the project folder, independently of the
current terminal directory. All fractions must share the same dose grid.
"""

import copy
import csv
import sys
from pathlib import Path

import numpy as np
import pydicom
from pydicom.uid import UID
from scipy.ndimage import distance_transform_edt, map_coordinates


# ======================== USER CONFIGURATION ===========================
# Read every .npy fraction in this folder; list individual files if needed.
DOSE_INPUTS = [Path("DoseMatrix")]
# RayStation numeric arrays in this project represent cGy; 0.01 converts to Gy.
NPY_VALUE_SCALE = 0.01
RTSS_PATH = Path("RS.dcm")  # Used only for DICOM-based ROI metrics.
STRUCTURE_NAME = "PTV"  # Case-insensitive substring; use None to list ROIs.
LIST_STRUCTURES_ONLY = False
PRESCRIPTION_DOSE_GY = None  # Example: 60.0; required for Rx-based metrics.
ABSOLUTE_VOLUME_DOSES_GY = [5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
BODY_STRUCTURE_NAME = None  # Example: "Body"; required for CI.
REFERENCE_ISODOSE_PERCENT = 95.0  # Isodose level used by CI and coverage.
MASK_SUPERSAMPLING = 8  # In-plane relative-volume samples per voxel edge.

# Set a path to export results, or None to skip that output.
SUMMED_DOSE_OUTPUT = None  # Example: Path("RD_total.dcm")
METRICS_CSV_OUTPUT = None  # Example: Path("metrics.csv")
# =======================================================================


# ----------------------------------------------------------------------
# 1. NumPy fraction dose loading
# ----------------------------------------------------------------------
def find_npy_files(paths):
    """Collect unique NumPy dose arrays from files or directory paths."""
    files = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            files.extend(sorted(path.glob("*.npy")))
        elif path.is_file():
            if path.suffix.lower() != ".npy":
                sys.exit(f"Unsupported NumPy dose input: {path}")
            files.append(path)
        else:
            sys.exit(f"Path not found: {path}")

    unique_files = []
    seen_paths = set()
    for file_path in files:
        resolved_path = file_path.resolve()
        if resolved_path not in seen_paths:
            unique_files.append(file_path)
            seen_paths.add(resolved_path)
    if not unique_files:
        sys.exit("No NumPy dose files found in the specified paths.")
    return unique_files


def load_npy_dose(file_path, value_scale=NPY_VALUE_SCALE):
    """Load a 3D RayStation NumPy dose array and convert its values to Gy."""
    file_path = Path(file_path)
    try:
        dose = np.load(file_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{file_path.name}: Cannot read NumPy dose array: {exc}") from exc
    if dose.ndim != 3 or 0 in dose.shape:
        sys.exit(f"{file_path.name}: Expected a non-empty 3D array, got {dose.shape}.")
    if not np.issubdtype(dose.dtype, np.number) or np.issubdtype(dose.dtype, np.complexfloating):
        sys.exit(f"{file_path.name}: Expected real numeric dose values, got {dose.dtype}.")
    if not np.isfinite(value_scale) or value_scale <= 0:
        raise ValueError("NPY_VALUE_SCALE must be positive.")
    if not np.all(np.isfinite(dose)):
        sys.exit(f"{file_path.name}: Dose values must all be finite.")
    if np.any(dose < 0):
        sys.exit(f"{file_path.name}: Dose values must be non-negative.")
    return dose.astype(np.float64) * value_scale


def sum_npy_fractions(paths, value_scale=NPY_VALUE_SCALE):
    """Sum NumPy fraction doses in Gy without changing their axis order."""
    total = None
    expected_shape = None
    for file_path in paths:
        file_path = Path(file_path)
        dose = load_npy_dose(file_path, value_scale)
        if total is None:
            total, expected_shape = dose, dose.shape
            print(f"  [Reference array] {file_path.name}  Shape: {dose.shape}")
        elif dose.shape != expected_shape:
            sys.exit(f"{file_path.name}: Dose shape {dose.shape} does not match "
                     f"the reference shape {expected_shape}.")
        else:
            total += dose
            print(f"  [Summed]           {file_path.name}")
        print(f"    Dose range: {dose.min():.6g} to {dose.max():.6g} Gy")
    return total


def print_dose_summary(dose_array):
    """Report basic numerical properties of an exported dose matrix."""
    nonzero_doses = dose_array[dose_array != 0]
    print(f"  Array shape: {dose_array.shape}")
    print(f"  Array dtype: {dose_array.dtype}")
    print(f"  Minimum dose: {dose_array.min():.6g} Gy")
    print(f"  Maximum dose: {dose_array.max():.6g} Gy")
    print(f"  Mean dose (entire matrix): {dose_array.mean():.6g} Gy")
    print("  Median non-zero dose: "
          f"{np.median(nonzero_doses):.6g} Gy" if nonzero_doses.size
          else "  Median non-zero dose: undefined (all voxels are zero)")


# ----------------------------------------------------------------------
# 2. DICOM input discovery and geometry checks
# ----------------------------------------------------------------------
def find_rtdose_files(paths):
    """Collect unique RT Dose files from the specified paths."""
    files = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            files.extend(sorted(path.iterdir()))
        elif path.is_file():
            files.append(path)
        else:
            sys.exit(f"Path not found: {path}")

    rtdose_files = []
    seen_paths = set()
    for file_path in files:
        try:
            resolved_path = file_path.resolve()
            if resolved_path in seen_paths:
                continue
            ds = pydicom.dcmread(file_path, stop_before_pixels=True)
        except (OSError, pydicom.errors.InvalidDicomError):
            continue  # Skip non-DICOM files.
        if getattr(ds, "Modality", "") == "RTDOSE":
            rtdose_files.append(file_path)
            seen_paths.add(resolved_path)
    if not rtdose_files:
        sys.exit("No RT Dose files found in the specified paths.")
    return rtdose_files


def check_geometry(ref, ds, name):
    """Exit if a dose grid differs from the reference grid."""
    def get(d, tag, default=None):
        return getattr(d, tag, default)

    def plan_uid(d):
        plans = get(d, "ReferencedRTPlanSequence", [])
        return get(plans[0], "ReferencedSOPInstanceUID") if plans else None

    for tag in ("GridFrameOffsetVector", "ReferencedRTPlanSequence"):
        if tag not in ref or tag not in ds:
            sys.exit(f"Dose grid mismatch ({tag}): required metadata missing.")

    checks = [
        ("Rows / Columns", (get(ref, "Rows"), get(ref, "Columns")),
                           (get(ds, "Rows"), get(ds, "Columns"))),
        ("NumberOfFrames", get(ref, "NumberOfFrames", 1),
                           get(ds, "NumberOfFrames", 1)),
        ("ImagePositionPatient", list(get(ref, "ImagePositionPatient", [])),
                                 list(get(ds, "ImagePositionPatient", []))),
        ("PixelSpacing", list(get(ref, "PixelSpacing", [])),
                         list(get(ds, "PixelSpacing", []))),
        ("ImageOrientationPatient", [round(float(v), 8) for v in get(ref, "ImageOrientationPatient", [])],
                                    [round(float(v), 8) for v in get(ds, "ImageOrientationPatient", [])]),
        ("GridFrameOffsetVector", [round(float(v), 4) for v in get(ref, "GridFrameOffsetVector", [])],
                                  [round(float(v), 4) for v in get(ds, "GridFrameOffsetVector", [])]),
        ("PatientID", get(ref, "PatientID"), get(ds, "PatientID")),
        ("PatientName", str(get(ref, "PatientName", "")),
                     str(get(ds, "PatientName", ""))),
        ("PatientBirthDate", get(ref, "PatientBirthDate", ""),
                            get(ds, "PatientBirthDate", "")),
        ("PatientSex", get(ref, "PatientSex", ""),
                       get(ds, "PatientSex", "")),
        ("FrameOfReferenceUID", get(ref, "FrameOfReferenceUID"),
                               get(ds, "FrameOfReferenceUID")),
        ("StudyInstanceUID", get(ref, "StudyInstanceUID"),
                             get(ds, "StudyInstanceUID")),
        ("ReferencedRTPlanUID", plan_uid(ref), plan_uid(ds)),
        ("DoseType", str(get(ref, "DoseType", "")).upper(),
                     str(get(ds, "DoseType", "")).upper()),
    ]
    for label, a, b in checks:
        if a != b:
            sys.exit(f"Dose grid mismatch ({label}):\n  Reference: {a}\n  {name}: {b}\n"
                     "Resample all doses onto the same grid in the TPS before summation.")


def validate_fraction_metadata(ds, name):
    """Reject non-fraction doses to avoid double counting."""
    dose_type = str(getattr(ds, "DoseType", "")).upper()
    if dose_type != "PHYSICAL":
        sys.exit(f"{name}: Expected physical dose (DoseType = PHYSICAL), got "
                 f"{dose_type or 'missing'}. RBE-weighted or EQD2 dose requires "
                 "a separate validated calculation.")
    summation_type = getattr(ds, "DoseSummationType", "").upper()
    if summation_type != "FRACTION":
        sys.exit(f"{name}: Expected DoseSummationType = FRACTION, got "
                 f"{summation_type or 'missing'}. BEAM and PLAN doses may "
                 "already include multiple fractions and must not be summed blindly.")


# ----------------------------------------------------------------------
# 2. Cumulative dose calculation
# ----------------------------------------------------------------------
def sum_fractions(files):
    """Sum fraction doses in Gy and return the total and reference dataset."""
    total = None
    ref = None
    seen_uids = set()
    for file_path in files:
        ds = pydicom.dcmread(file_path)
        for tag in ("DoseUnits", "DoseGridScaling", "ImagePositionPatient",
                    "ImageOrientationPatient", "PixelSpacing", "Rows",
                    "Columns", "NumberOfFrames", "FrameOfReferenceUID",
                    "GridFrameOffsetVector", "ReferencedRTPlanSequence",
                    "PatientID", "StudyInstanceUID", "SOPClassUID"):
            if tag not in ds:
                sys.exit(f"{file_path.name}: Missing required RT Dose tag {tag}.")
        if str(ds.DoseUnits).upper() != "GY":
            sys.exit(f"{file_path.name}: Unsupported DoseUnits = "
                     f"{ds.DoseUnits}; expected GY.")
        try:
            scaling = float(ds.DoseGridScaling)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{file_path.name}: Invalid DoseGridScaling.") from exc
        if not np.isfinite(scaling) or scaling <= 0:
            sys.exit(f"{file_path.name}: DoseGridScaling must be positive.")
        if UID(ds.SOPClassUID) != UID("1.2.840.10008.5.1.4.1.1.481.2"):
            sys.exit(f"{file_path.name}: Expected RT Dose Storage.")
        transfer_syntax = getattr(getattr(ds, "file_meta", None),
                                 "TransferSyntaxUID", None)
        if transfer_syntax is None:
            sys.exit(f"{file_path.name}: Missing DICOM transfer syntax.")
        if not UID(transfer_syntax).is_little_endian:
            sys.exit(f"{file_path.name}: Big-endian RT Dose is not supported.")
        validate_fraction_metadata(ds, file_path.name)
        sop_uid = getattr(ds, "SOPInstanceUID", None)
        if not sop_uid:
            sys.exit(f"{file_path.name}: Missing SOPInstanceUID.")
        if sop_uid in seen_uids:
            sys.exit(f"Duplicate RT Dose instance: {file_path.name}")
        seen_uids.add(sop_uid)
        if int(getattr(ds, "PixelRepresentation", 0)) != 0:
            sys.exit(f"{file_path.name}: Signed dose pixels are not supported.")
        dose = ds.pixel_array.astype(np.float64) * scaling
        expected_shape = (int(ds.NumberOfFrames), int(ds.Rows), int(ds.Columns))
        if dose.shape != expected_shape:
            sys.exit(f"{file_path.name}: Expected dose shape {expected_shape}, got {dose.shape}.")
        if not np.all(np.isfinite(dose)) or np.any(dose < 0):
            sys.exit(f"{file_path.name}: Dose values must be finite and non-negative.")
        if total is None:
            total, ref = dose, ds
            print(f"  [Reference grid] {file_path.name}  Shape: {dose.shape}")
        else:
            check_geometry(ref, ds, file_path.name)
            total += dose
            print(f"  [Summed]        {file_path.name}")
    return total, ref


def build_summed_dataset(total, ref):
    """Create a 32-bit cumulative RT Dose with a new SOP Instance UID."""
    if total.ndim != 3 or total.size == 0 or not np.all(np.isfinite(total)):
        raise ValueError("The cumulative dose must be a finite 3D array.")
    summed = copy.deepcopy(ref)
    new_scaling = max(float(total.max()) / 4_000_000_000.0, 1e-9)
    encoded = np.rint(total / new_scaling)
    if encoded.max() > np.iinfo(np.uint32).max:
        raise ValueError("The cumulative dose exceeds the RT Dose storage range.")
    summed.DoseGridScaling = new_scaling
    summed.PixelData = encoded.astype(np.uint32).tobytes()
    summed.BitsAllocated = 32
    summed.BitsStored = 32
    summed.HighBit = 31
    summed.PixelRepresentation = 0
    summed.DoseSummationType = "PLAN"
    summed.SeriesInstanceUID = pydicom.uid.generate_uid()
    # Fraction-specific references are not valid for a cumulative PLAN dose.
    for tag in ("ReferencedFractionGroupSequence", "ReferencedBeamSequence",
                "ReferencedBrachyApplicationSetupSequence"):
        if tag in summed:
            delattr(summed, tag)
    if hasattr(summed, "is_little_endian"):
        summed.is_little_endian = True
    # Keep dataset and file-meta identifiers consistent.
    summed.SOPInstanceUID = pydicom.uid.generate_uid()
    if not hasattr(summed, "file_meta"):
        summed.file_meta = pydicom.dataset.FileMetaDataset()
    summed.file_meta.MediaStorageSOPInstanceUID = summed.SOPInstanceUID
    summed.file_meta.MediaStorageSOPClassUID = summed.SOPClassUID
    summed.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    if not UID(summed.SOPClassUID).name == "RT Dose Storage":
        raise ValueError("The reference dataset is not an RT Dose object.")
    return summed


# ----------------------------------------------------------------------
# 3. Structure selection and cumulative DVH metrics
# ----------------------------------------------------------------------
def get_structures(rtss_ds):
    """Return ROI names indexed by their DICOM ROI numbers."""
    if "StructureSetROISequence" not in rtss_ds:
        sys.exit("No structures found in the RT Structure Set.")
    return {
        int(item.ROINumber): {"name": str(item.ROIName)}
        for item in rtss_ds.StructureSetROISequence
    }


def match_structure(structures, keyword):
    """Resolve a unique case-insensitive structure-name substring."""
    if not keyword:
        sys.exit("A structure name is required in USER CONFIGURATION.")
    hits = {
        number: item for number, item in structures.items()
        if keyword.upper() in item["name"].upper()
    }
    names = ", ".join(item["name"] for item in structures.values())
    if not hits:
        sys.exit(f"No structure name contains '{keyword}'. Available: {names}")
    if len(hits) > 1:
        matches = ", ".join(f"{k}:{v['name']}" for k, v in hits.items())
        sys.exit(f"'{keyword}' matches multiple structures: {matches}")
    return next(iter(hits))


def get_roi_contours(rtss_ds, roi):
    """Read planar polygon contours for one ROI."""
    for roi_contour in rtss_ds.get("ROIContourSequence", []):
        if int(roi_contour.ReferencedROINumber) == roi:
            return [
                np.asarray(contour.ContourData, dtype=float).reshape(-1, 3)
                for contour in roi_contour.get("ContourSequence", [])
                if getattr(contour, "ContourGeometricType", "") == "CLOSED_PLANAR"
                and int(contour.NumberOfContourPoints) >= 3
            ]
    return []


def _planar_coverage(polygons, grid_shape, row_spacing, column_spacing,
                     supersampling):
    """Rasterize same-plane contours and return relative coverage and distance."""
    margin = 2
    bounds = np.vstack(polygons)
    lower = np.floor(bounds.min(axis=0)).astype(int) - margin
    upper = np.ceil(bounds.max(axis=0)).astype(int) + margin
    if np.any(upper <= lower):
        return np.zeros(grid_shape), None, lower, upper
    local_shape = upper - lower
    sampling = (row_spacing / supersampling, column_spacing / supersampling)
    binary_mask = np.zeros(local_shape * supersampling, dtype=bool)
    for polygon in polygons:
        # Convert center-based DICOM indices to subpixel edge coordinates.
        # Subpixel q is sampled at q + 0.5 in this coordinate system.
        local_polygon = polygon - lower
        vertices = (local_polygon + 0.5) * supersampling
        rows, columns = np.mgrid[:local_shape[0] * supersampling,
                                :local_shape[1] * supersampling]
        rows = rows + 0.5
        columns = columns + 0.5
        inside = np.zeros(local_shape * supersampling, dtype=bool)
        for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
            if start[0] == end[0]:
                continue
            crosses = (start[0] > rows) != (end[0] > rows)
            intersection = start[1] + ((rows - start[0]) * (end[1] - start[1])
                                      / (end[0] - start[0]))
            inside ^= crosses & (columns < intersection)
        binary_mask ^= inside
    if not np.any(binary_mask):
        return np.zeros(grid_shape), None, lower, upper
    coverage = binary_mask.reshape(
        local_shape[0], supersampling, local_shape[1], supersampling,
    ).mean(axis=(1, 3))
    distance_pad = 0
    if np.all(binary_mask):
        # Use exterior padding only for the signed-distance field.
        binary_mask = np.pad(binary_mask, 1)
        distance_pad = 1
    # The distance sign is used for shape-based interpolation between contours.
    signed_distance = (
        distance_transform_edt(binary_mask, sampling=sampling)
        - distance_transform_edt(~binary_mask, sampling=sampling)
    )
    result = np.zeros(grid_shape)
    output_lower = np.maximum(0, lower)
    output_upper = np.minimum(grid_shape, upper)
    if np.any(output_upper > output_lower):
        source_lower = output_lower - lower
        source_upper = source_lower + output_upper - output_lower
        result[output_lower[0]:output_upper[0],
               output_lower[1]:output_upper[1]] = coverage[
                   source_lower[0]:source_upper[0],
                   source_lower[1]:source_upper[1],
               ]
    return result, signed_distance, lower - distance_pad, upper + distance_pad


def _distance_samples(field, bounds, grid_shape, row_spacing, column_spacing,
                      supersampling):
    """Evaluate signed distance at all dose-grid subpixel centers."""
    del row_spacing, column_spacing
    high_resolution_shape = np.asarray(grid_shape) * supersampling
    if field is None:
        return np.full(high_resolution_shape, -1.0)
    coordinates = np.mgrid[:high_resolution_shape[0], :high_resolution_shape[1]]
    samples = map_coordinates(
        np.asarray(field, dtype=np.float32),
        [coordinates[0] - bounds[0, 0] * supersampling,
         coordinates[1] - bounds[0, 1] * supersampling],
        order=1, mode="constant", cval=-1.0, prefilter=False,
    )
    return samples


def rasterize_roi(rtss_ds, summed_ds, roi):
    """Reconstruct ROI relative volumes on the exported dose grid."""
    polygons = get_roi_contours(rtss_ds, roi)
    if not polygons:
        sys.exit(f"ROI {roi} has no supported planar contours.")
    for item in rtss_ds.StructureSetROISequence:
        if int(item.ROINumber) == roi:
            frame_uid = getattr(item, "ReferencedFrameOfReferenceUID", None)
            if frame_uid and frame_uid != summed_ds.FrameOfReferenceUID:
                sys.exit(f"ROI {roi} belongs to a different frame of reference.")
            break
    if int(MASK_SUPERSAMPLING) != MASK_SUPERSAMPLING or MASK_SUPERSAMPLING < 1:
        raise ValueError("MASK_SUPERSAMPLING must be a positive integer.")
    supersampling = int(MASK_SUPERSAMPLING)
    orientation = np.asarray(summed_ds.ImageOrientationPatient, dtype=float)
    if (orientation.shape != (6,) or not np.all(np.isfinite(orientation))
            or abs(float(orientation[:3] @ orientation[3:])) > 1e-6
            or not np.allclose(np.linalg.norm(orientation.reshape(2, 3), axis=1),
                               1.0, atol=1e-6)):
        sys.exit("Invalid ImageOrientationPatient in the dose grid.")
    row_cosines, column_cosines = orientation[:3], orientation[3:]
    normal = np.cross(row_cosines, column_cosines)
    if np.any(np.abs(normal) > 1e-6) and not np.allclose(
            np.abs(normal), [0, 0, 1], atol=1e-6):
        sys.exit("Only axial dose grids are supported for ROI metrics.")
    origin = np.asarray(summed_ds.ImagePositionPatient, dtype=float)
    row_spacing, column_spacing = map(float, summed_ds.PixelSpacing)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        sys.exit("Invalid ImagePositionPatient in the dose grid.")
    if not np.isfinite([row_spacing, column_spacing]).all() or min(
            row_spacing, column_spacing) <= 0:
        sys.exit("PixelSpacing must be positive.")
    grid_shape = (int(summed_ds.Rows), int(summed_ds.Columns))
    offsets = np.asarray(summed_ds.GridFrameOffsetVector, dtype=float)
    frame_count = int(summed_ds.NumberOfFrames)
    if (offsets.size != frame_count or not np.all(np.isfinite(offsets))
            or (frame_count > 1 and not np.allclose(
                np.diff(offsets), np.diff(offsets)[0], atol=1e-5))):
        sys.exit("DVH metrics require finite, uniformly spaced dose planes.")
    if frame_count > 1:
        slice_thickness = float(abs(np.diff(offsets)[0]))
    elif "SliceThickness" in summed_ds and float(summed_ds.SliceThickness) > 0:
        slice_thickness = float(summed_ds.SliceThickness)
    else:
        sys.exit("Single-frame dose requires a positive SliceThickness.")
    contour_groups = {}
    for polygon in polygons:
        if not np.all(np.isfinite(polygon)):
            sys.exit(f"ROI {roi} has non-finite contour coordinates.")
        position = float(np.median(polygon @ normal))
        if not np.allclose(polygon @ normal, position, atol=1e-3):
            sys.exit(f"ROI {roi} contains a non-planar contour.")
        contour_groups.setdefault(position, []).append(polygon)
    contour_groups = {
        position: [np.column_stack((
            (group_polygons - origin) @ column_cosines / row_spacing,
            (group_polygons - origin) @ row_cosines / column_spacing,
        )) for group_polygons in group]
        for position, group in contour_groups.items()
    }
    positions = sorted(contour_groups)
    relative_positions = ((np.asarray(positions) - origin @ normal)
                          / (normal @ normal))
    contour_covered = (
        (relative_positions >= offsets.min() - slice_thickness / 2)
        & (relative_positions <= offsets.max() + slice_thickness / 2)
    )
    if not np.all(contour_covered):
        print("  [Warning] ROI extends beyond the dose grid. Only covered "
              "voxels contribute to DVH statistics, as in RayStation.")
    plane_positions = ((origin[None, :] + offsets[:, None] * normal[None, :])
                       @ normal)
    planes = {}
    for frame, position in enumerate(plane_positions):
        if position < positions[0] - 1e-3 or position > positions[-1] + 1e-3:
            continue
        lower = [value for value in positions if value <= position + 1e-3]
        upper = [value for value in positions if value >= position - 1e-3]
        planes[frame] = (lower[-1], upper[0], position)
    if not planes:
        sys.exit("The selected ROI has no contours inside the dose grid.")
    mask = np.zeros((grid_shape[0], grid_shape[1], frame_count), dtype=np.float32)
    planar_data = {}
    for frame, (lower_position, upper_position, position) in planes.items():
        if lower_position not in planar_data:
            planar_data[lower_position] = _planar_coverage(
                contour_groups[lower_position], grid_shape,
                row_spacing, column_spacing, supersampling)
        if upper_position not in planar_data:
            planar_data[upper_position] = _planar_coverage(
                contour_groups[upper_position], grid_shape,
                row_spacing, column_spacing, supersampling)
        lower_data = planar_data[lower_position]
        if lower_position == upper_position:
            mask[:, :, frame] = lower_data[0]
            continue
        weight = (position - lower_position) / (upper_position - lower_position)
        upper_data = planar_data[upper_position]
        lower_bounds = np.array([lower_data[2], lower_data[3]])
        upper_bounds = np.array([upper_data[2], upper_data[3]])
        bounds = np.column_stack((
            np.minimum(lower_bounds[:, 0], upper_bounds[:, 0]),
            np.maximum(lower_bounds[:, 1], upper_bounds[:, 1]),
        ))
        lower_distance = _distance_samples(
            lower_data[1], lower_bounds, grid_shape,
            row_spacing, column_spacing, supersampling)
        upper_distance = _distance_samples(
            upper_data[1], upper_bounds, grid_shape,
            row_spacing, column_spacing, supersampling)
        interpolated = ((1.0 - weight) * lower_distance + weight * upper_distance)
        # Sample both distance fields on the same high-resolution dose grid.
        mask[:, :, frame] = (interpolated >= 0.0).reshape(
            grid_shape[0], supersampling, grid_shape[1], supersampling,
        ).mean(axis=(1, 3))
    if not np.any(mask):
        sys.exit("The selected ROI has no voxels inside the dose grid.")
    voxel_volume_cc = row_spacing * column_spacing * slice_thickness / 1000.0
    return mask.transpose(2, 0, 1), voxel_volume_cc


def dose_at_volume(dose_values, voxel_weights, volume_cc):
    """Return the discrete minimum dose accumulated by the hottest volume."""
    dose_values = np.asarray(dose_values, dtype=float)
    voxel_weights = np.asarray(voxel_weights, dtype=float)
    if dose_values.ndim != 1 or dose_values.shape != voxel_weights.shape:
        raise ValueError("Dose values and voxel weights must be aligned 1D arrays.")
    if (not np.all(np.isfinite(dose_values))
            or not np.all(np.isfinite(voxel_weights))
            or np.any(voxel_weights < 0) or np.sum(voxel_weights) <= 0):
        raise ValueError("Dose values must be finite and voxel weights positive.")
    total_volume_cc = np.sum(voxel_weights)
    if not np.isfinite(volume_cc) or volume_cc <= 0:
        raise ValueError("Volume must be positive.")
    if volume_cc > total_volume_cc + 1e-9:
        return float("nan")
    order = np.argsort(-dose_values, kind="stable")
    cumulative_volume_cc = np.cumsum(voxel_weights[order])
    tolerance = max(1e-9, total_volume_cc * 1e-9)
    index = int(np.searchsorted(
        cumulative_volume_cc + tolerance, volume_cc, side="left"))
    return float(dose_values[order[min(index, len(order) - 1)]])


def volume_receiving_dose(dose_values, voxel_weights, dose_gy):
    """Return the volume and percentage receiving at least the requested dose."""
    tolerance = max(1e-12, abs(dose_gy) * 1e-12)
    volume_cc = float(np.sum(voxel_weights[dose_values >= dose_gy - tolerance]))
    return volume_cc, 100.0 * volume_cc / np.sum(voxel_weights)


def compute_metrics(dose_array, target_mask, voxel_volume_cc,
                    prescription_gy=None, absolute_doses_gy=(),
                    reference_mask=None, isodose_percent=95.0):
    """Calculate RayStation-style statistics without DVH interpolation."""
    dose_array = np.asarray(dose_array, dtype=float)
    target_mask = np.asarray(target_mask, dtype=float)
    if dose_array.shape != target_mask.shape:
        raise ValueError("Dose and target masks must share a grid.")
    if not np.isfinite(voxel_volume_cc) or voxel_volume_cc <= 0:
        raise ValueError("Voxel volume must be positive.")
    if (not np.all(np.isfinite(target_mask)) or np.any(target_mask < 0)
            or np.any(target_mask > 1)):
        raise ValueError("ROI relative volumes must be between zero and one.")
    target_voxels = target_mask > 0
    target_doses = dose_array[target_voxels]
    target_weights = target_mask[target_voxels] * voxel_volume_cc
    if target_doses.size == 0:
        raise ValueError("The target mask is empty.")
    if not np.all(np.isfinite(target_doses)):
        raise ValueError("Target doses must be finite.")
    target_volume_cc = float(np.sum(target_weights))
    metrics = {"Volume_cc": target_volume_cc}

    for percent in (98, 95, 50, 5, 2):
        metrics[f"D{percent}_Gy"] = dose_at_volume(
            target_doses, target_weights, target_volume_cc * percent / 100.0)
    for volume_cc in (0.03, 2.0):
        metrics[f"D{volume_cc:g}cc_Gy"] = dose_at_volume(
            target_doses, target_weights, volume_cc)
    mean_dose = float(np.sum(target_doses * target_weights) / target_volume_cc)
    variance = float(np.sum(
        (target_doses - mean_dose) ** 2 * target_weights) / target_volume_cc)
    metrics["Dmean_Gy"] = mean_dose
    metrics["Dstd_Gy"] = np.sqrt(max(0.0, variance))
    metrics["Dmin_Gy"] = float(np.min(target_doses))
    metrics["Dmax_Gy"] = float(np.max(target_doses))
    metrics["HI_ICRU83"] = (
        (metrics["D2_Gy"] - metrics["D98_Gy"]) / metrics["D50_Gy"]
        if metrics["D50_Gy"] > 0 else float("nan")
    )
    for dose_gy in absolute_doses_gy:
        if not np.isfinite(dose_gy) or dose_gy < 0:
            raise ValueError("Absolute volume-dose thresholds must be non-negative.")
        volume_cc, volume_percent = volume_receiving_dose(
            target_doses, target_weights, dose_gy)
        metrics[f"V{dose_gy:g}Gy_cc"] = volume_cc
        metrics[f"V{dose_gy:g}Gy_percent"] = volume_percent

    if prescription_gy is not None:
        if not np.isfinite(prescription_gy) or prescription_gy <= 0:
            raise ValueError("Prescription dose must be positive.")
        if reference_mask is not None and not 0 < isodose_percent <= 100:
            raise ValueError("Reference isodose must be in (0, 100] percent.")
        for percent in (90, 95, 100, 105, 107, 110):
            dose_gy = prescription_gy * percent / 100.0
            _, volume_percent = volume_receiving_dose(
                target_doses, target_weights, dose_gy)
            metrics[f"V{percent}percentRx_percent"] = volume_percent
        metrics["HI_D2minusD98_over_Rx"] = (
            metrics["D2_Gy"] - metrics["D98_Gy"]) / prescription_gy

        if reference_mask is not None:
            reference_mask = np.asarray(reference_mask, dtype=float)
            if reference_mask.shape != target_mask.shape:
                raise ValueError("Reference and target masks must share a grid.")
            if (not np.all(np.isfinite(reference_mask))
                    or np.any(reference_mask < 0) or np.any(reference_mask > 1)):
                raise ValueError("Reference ROI relative volumes must be in [0, 1].")
            evaluation_weights = np.maximum(reference_mask, target_mask)
            isodose_gy = prescription_gy * isodose_percent / 100.0
            isodose_tolerance = max(1e-12, isodose_gy * 1e-12)
            isodose_voxels = dose_array >= isodose_gy - isodose_tolerance
            target_isodose_cc = float(np.sum(
                target_mask[isodose_voxels]) * voxel_volume_cc)
            reference_isodose_cc = float(np.sum(
                evaluation_weights[isodose_voxels]) * voxel_volume_cc)
            if reference_isodose_cc > 0:
                metrics[f"CI_Paddick_{isodose_gy:g}Gy"] = (
                    target_isodose_cc ** 2
                    / (target_volume_cc * reference_isodose_cc)
                )
                metrics[f"CI_RTOG_{isodose_gy:g}Gy"] = (
                    reference_isodose_cc / target_volume_cc
                )
            else:
                metrics[f"CI_Paddick_{isodose_gy:g}Gy"] = float("nan")
                metrics[f"CI_RTOG_{isodose_gy:g}Gy"] = float("nan")
    return metrics


# ----------------------------------------------------------------------
# Cursor run workflow
# ----------------------------------------------------------------------
def resolve_path(path):
    """Resolve a configured path relative to the project folder."""
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path.resolve()


def main():
    dose_inputs = [resolve_path(path) for path in DOSE_INPUTS]
    rtss_path = resolve_path(RTSS_PATH)
    summed_output = resolve_path(SUMMED_DOSE_OUTPUT) if SUMMED_DOSE_OUTPUT else None
    csv_output = resolve_path(METRICS_CSV_OUTPUT) if METRICS_CSV_OUTPUT else None
    is_npy_input = any(
        path.is_file() and path.suffix.lower() == ".npy"
        or path.is_dir() and any(path.glob("*.npy"))
        for path in dose_inputs
    )

    if is_npy_input:
        print("== NumPy fraction dose validation ==")
        files = find_npy_files(dose_inputs)
        print(f"Found {len(files)} NumPy fraction dose file(s).")
        total = sum_npy_fractions(files)
        print_dose_summary(total)
        if len(files) > 1:
            print(f"Cumulative dose from {len(files)} fractions: "
                  f"{total.min():.6g} to {total.max():.6g} Gy")
        if any((summed_output, csv_output, LIST_STRUCTURES_ONLY)):
            sys.exit("DICOM output, ROI metrics, and structure listing are not "
                     "available for NumPy doses without exported grid metadata.")
        print("NumPy dose validation passed. ROI-based DVH metrics require a "
              "DICOM RT Dose grid and a matching RT Structure Set.")
        return

    print("== 1. Collect RT Dose files ==")
    files = find_rtdose_files(dose_inputs)
    print(f"Found {len(files)} fraction dose files.")

    print("\n== 2. Sum fraction doses ==")
    total, ref = sum_fractions(files)
    print(f"Cumulative dose range: {total.min():.3f} to {total.max():.3f} Gy")

    print("\n== 3. Load structure set ==")
    if not rtss_path.is_file():
        sys.exit(f"RT Structure Set file not found: {rtss_path}")
    rtss_ds = pydicom.dcmread(rtss_path)
    if getattr(rtss_ds, "Modality", "") != "RTSTRUCT":
        sys.exit(f"Expected an RT Structure Set: {rtss_path}")
    for tag in ("PatientID", "StudyInstanceUID", "FrameOfReferenceUID"):
        if tag not in rtss_ds:
            sys.exit(f"RT Structure Set is missing required tag {tag}.")
        if str(getattr(rtss_ds, tag)) != str(getattr(ref, tag)):
            sys.exit(f"RTSS/dose mismatch ({tag}): {getattr(rtss_ds, tag)} "
                     f"!= {getattr(ref, tag)}")
    referenced_frames = [
        str(frame.FrameOfReferenceUID)
        for frame in rtss_ds.get("ReferencedFrameOfReferenceSequence", [])
        if "FrameOfReferenceUID" in frame
    ]
    if (referenced_frames and "FrameOfReferenceUID" in ref
            and ref.FrameOfReferenceUID not in referenced_frames):
        sys.exit("The RTSS does not reference the dose frame of reference.")
    structs = get_structures(rtss_ds)
    if LIST_STRUCTURES_ONLY or STRUCTURE_NAME is None:
        print("Structures (ROI number: name):")
        for number, item in structs.items():
            print(f"  {number}: {item['name']}")
        if LIST_STRUCTURES_ONLY:
            return
        sys.exit("\nSet STRUCTURE_NAME in USER CONFIGURATION and run again.")
    roi = match_structure(structs, STRUCTURE_NAME)
    name = structs[roi]["name"]
    print(f"Target: {name} (ROI #{roi})")

    print("\n== 4. Calculate DVH metrics ==")
    summed = build_summed_dataset(total, ref)
    target_mask, voxel_volume_cc = rasterize_roi(rtss_ds, summed, roi)
    reference_mask = None
    if BODY_STRUCTURE_NAME is not None:
        if PRESCRIPTION_DOSE_GY is None:
            sys.exit("Set PRESCRIPTION_DOSE_GY before calculating CI.")
        body_roi = match_structure(structs, BODY_STRUCTURE_NAME)
        reference_mask, body_voxel_volume_cc = rasterize_roi(
            rtss_ds, summed, body_roi)
        if not np.isclose(body_voxel_volume_cc, voxel_volume_cc):
            sys.exit("Target and reference masks have different voxel volumes.")
        print(f"CI reference contour: {structs[body_roi]['name']}")
    metrics = compute_metrics(
        total, target_mask, voxel_volume_cc,
        prescription_gy=PRESCRIPTION_DOSE_GY,
        absolute_doses_gy=ABSOLUTE_VOLUME_DOSES_GY,
        reference_mask=reference_mask,
        isodose_percent=REFERENCE_ISODOSE_PERCENT,
    )
    metrics = {"Structure": name, "FractionCount": len(files), **metrics}
    for key, value in metrics.items():
        print(f"  {key}: {value:.6g}" if isinstance(value, float)
              else f"  {key}: {value}")

    if summed_output:
        summed_output.parent.mkdir(parents=True, exist_ok=True)
        pydicom.dcmwrite(summed_output, summed, write_like_original=False)
        print(f"\nCumulative dose saved to: {summed_output}")

    if csv_output:
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_output, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(metrics.keys())
            writer.writerow([
                f"{value:.10g}" if isinstance(value, (float, np.floating)) else value
                for value in metrics.values()
            ])
        print(f"Metrics saved to: {csv_output}")


if __name__ == "__main__":
    main()
