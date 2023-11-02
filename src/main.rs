#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
#[global_allocator]
static ALLOC: snmalloc_rs::SnMalloc = snmalloc_rs::SnMalloc;
use anyhow::Result;
use dashmap::DashMap;
use dicom::{
    object::{open_file, FileDicomObject, InMemDicomObject, Tag},
};
use dicom_pixeldata::PixelDecoder;
use eframe::egui::{self,ColorImage,Image, TextureId, menu, Color32, Grid, RichText, SliderOrientation};
use ndarray::s;
use jwalk::{DirEntry, WalkDirGeneric};
use polars::prelude::*;
use rayon::prelude::*;
use std::collections::HashSet;
use std::{
    io::{Read,Cursor},
    cmp::Ordering,
    fs::File,
    path::{Path, PathBuf},
};
use csv::Writer;
use chrono::prelude::*;
use polars::series::ops::NullBehavior;

use ::zip::read::ZipArchive;
#[inline(always)]
fn read_csv_from_zip(path: &Path, csv_file_name: &str) -> Option<String> {
    let file = File::open(&path).ok()?;
    let mut archive = ZipArchive::new(file).ok()?;

    let mut file = archive.by_name(csv_file_name).ok()?;
    let mut contents = String::new();
    file.read_to_string(&mut contents).ok()?;
    Some(contents)
}
#[inline(always)]
fn get_image_position(obj: &InMemDicomObject) -> Option<f64> {
    obj.element(Tag(0x0020, 0x0032))
        .ok()
        .and_then(|e| e.to_multi_float64().ok())
        .map(|v| v[2])
}
#[inline(always)]
fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        active: true,
        centered: true,
        icon_data: Some(eframe::IconData::try_from_png_bytes(include_bytes!("sona.png")).unwrap()),
        always_on_top: false,
        maximized: false,
        decorated: true,
        drag_and_drop_support: true,
        initial_window_pos: None,
        initial_window_size: Some(egui::vec2(1024.0, 512.0)),
        min_window_size: None,
        max_window_size: Some(egui::vec2(2048.0, 2048.0)),
        resizable: true,
        transparent: true,
        vsync: true,
        ..Default::default()
    };

    eframe::run_native(
        "SonALAsense Parameter Tool",
        options,
        Box::new(|cc| Box::new(MyEguiApp::new(cc))),
    )
}
#[inline(always)]
fn is_dicom(entry: &DirEntry<(u32, bool)>) -> bool {
    entry.file_type().is_file() && open_file(entry.path()).is_ok()
}
#[inline(always)]
fn to_dicom(entry: DirEntry<(u32, bool)>) -> Option<(FileDicomObject<InMemDicomObject>, PathBuf)> {
    let path = entry.path().to_path_buf();
    open_file(path.clone()).ok().map(|object| (object, path))
}

#[inline(always)]
fn read_csv_file(path: &Path) -> PolarsResult<DataFrame> {
    let mut df = LazyCsvReader::new(path)
        .has_header(true)
        .with_encoding(CsvEncoding::Utf8)
        .with_try_parse_dates(true)
        .finish().expect("Not a csv")
        .with_columns([col("Time").str().to_datetime(
            Some(TimeUnit::Milliseconds),
            None,
            StrptimeOptions {
                format: Some("%Y%m%d%H%M%S ".to_string()),
                ..Default::default()
            },
            lit("raise"),
        )])
        .with_columns([
            col("Time").dt().year().alias("Year"),
            col("Time").dt().month().alias("Month"),
            col("Time").dt().day().alias("Day"),
            col("Time").dt().hour().alias("Hour"),
            col("Time").dt().minute().alias("Minute"),
            col("Time").dt().second().alias("Second"),
            (col("Energy[J]") / col("Num. of SubSonic")).alias("Energy per subspot"),
            (col("Num. of Pulses") * col("Pulse Duration")).alias("CumPulseDurperRep"),
            (col("Act. Energy[J]") / col("Num. of SubSonic")).alias("Act. Energy per subspot"),
            col("Target Volume [cc]").cumsum(false).alias("cum_vol"),
        ])
        .drop_nulls(Some(Vec::<Expr>::new()))
        .drop_columns(vec![
            "Energy[J]",
            "Protocol Name ",
            "Act. Energy[J]",
            "Frequency[Hz]",
            "Time",
            "Mode",
            "Treated Dose[cc]",
            "Stopped",
            "Target Cav.Dose",
            "Focal RAS-R",
            "Focal RAS-A",
            "Focal RAS-S",
        ])
        .collect().expect("failed to modify");
    // Prepend "simplify" to the file name
    let mut folder_name = path.parent().expect("There is no parent");
    let folder_name = folder_name.file_name().unwrap_or(path.file_name().unwrap());
    let parent_dir = path.parent().unwrap();
    let new_path = parent_dir.join(folder_name.to_str().unwrap().to_owned() + "_simplify.csv");

    // Create a new file with the new name
    let mut file = File::create(new_path).expect("could not create file");

    // Write the DataFrame to the csv file with header and comma separator
    CsvWriter::new(&mut file)
        .has_header(true)
        .with_float_precision(Some(2))
        .finish(&mut df)?;

    Ok(df)
}

#[inline(always)]
fn extract_zip(path: &Path, dir_path: &Path) {
    let reader = File::open(path).expect("Unable to open file");
    let mut archive = ZipArchive::new(reader).expect("Unable to create ZipArchive");

    let map = DashMap::new();

    for i in 0..archive.len() {
        let mut file = archive.by_index(i).expect("Unable to access file in archive");
        let outpath = dir_path.join(file.mangled_name().file_name().unwrap());

        if file.name().ends_with(".zip") {
            // If the file is a zip file, recursively extract it
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer).expect("Unable to read file");
            let reader = Cursor::new(buffer);
            let mut inner_archive = ZipArchive::new(reader).expect("Unable to create inner ZipArchive");

            for j in 0..inner_archive.len() {
                let mut inner_file = inner_archive.by_index(j).expect("Unable to access inner file in archive");
                let inner_outpath = dir_path.join(inner_file.mangled_name().file_name().unwrap());

                if inner_file.name().ends_with(".bmp") {
                    // If the inner file is a BMP image, extract it
                    let mut buffer = Vec::new();
                    inner_file.read_to_end(&mut buffer).expect("Unable to read inner file");
                    map.insert(inner_outpath, buffer);
                }
            }
        } else if file.name().ends_with(".bmp") {
            // If the outer file is a BMP image, extract it
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer).expect("Unable to read outer file");
            map.insert(outpath, buffer);
        }
    }

    // Write all files at once
    map.par_iter().for_each(|entry| {
        let (outpath, buffer) = entry.pair();
        std::fs::write(outpath, buffer).expect("Unable to write file");
    });
}
#[derive(Default)]
struct MyEguiApp {
    power: f32,
    subspots: i32,
    spacing: f64,
    pulsetrain: i32,
    pulseduration: f64,
    reptime: f64,
    cycles: i32,
    show_mode: Box<str>,
    filepath: Option<PathBuf>,
    df: Option<DataFrame>,
    selected_uid: String,
    summaryname: String,
    selected_folder: Option<PathBuf>,
    presorted: DashMap<String,Vec<(FileDicomObject<InMemDicomObject>, PathBuf)>>,
    current_image_index: usize,
    extract_images: bool,
    sonication_number: i32,
    grid_data: Vec<Vec<String>>,
}

impl MyEguiApp {
    #[inline(always)]
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        egui_extras::install_image_loaders(&cc.egui_ctx);
        Self {
            power: 10.0,
            subspots: 32,
            spacing: 3.0,
            pulsetrain: 10,
            pulseduration: 2.4,
            reptime: 1.00,
            cycles: 100,
            show_mode: "parameters".into(),
            filepath: None,
            df: None,
            selected_uid: String::new(),
            summaryname: "summary.csv".to_string(),
            selected_folder: None, // Add this line
            presorted: DashMap::new(), // Add this line
            current_image_index: 1,
            extract_images: true,
            sonication_number: 1,
            grid_data: vec![
                        vec![
                            "Son.\n (#)".to_string(),
                            "Time".to_string(),
                            "Power\n (W)".to_string(),
                            "foci\n (#)".to_string(),
                            "spacing\n (mm)".to_string(),
                            "Pulses\n (#)".to_string(),
                            "P.Dur\n (ms)".to_string(),
                            "Rep\nTime\n (s)".to_string(),
                            "Reps\n (#)".to_string(),
                            "En.foci\n (J/spot)".to_string(),
                            "Tar\n Vol\n (mm3)".to_string(),
                            "En.Vol\n (J/mm3)".to_string(),
                            "Son.Dur\n (s)".to_string(),
                            "PRF\n (Hz)".to_string(),
                            "Rec\nPhase\n (ms)".to_string(),
                            "Period\n (ms)".to_string(),
                            "DC\n (%)".to_string(),
                            "DCPS \n (%)".to_string(),
                        ],
                        ],
        }
    }
    #[inline(always)]
    fn show_summary_ui(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.with_layout(egui::Layout::top_down_justified(egui::Align::Center), |ui| {
                if ui.button(RichText::new("From CSV").size(25.0)).clicked() {
                    if let Some(path) = rfd::FileDialog::new().pick_file() {
                        self.filepath = Some(path);
                        self.df = Some(read_csv_file(self.filepath.as_ref().unwrap().as_path()).unwrap());
                    }
                };
            });
            ui.with_layout(egui::Layout::top_down_justified(egui::Align::Center), |ui| {

                if ui.button(RichText::new("From ZIP").size(25.0)).clicked() {
                    if let Some(path) = rfd::FileDialog::new().pick_file() {
                    let csv_contents = read_csv_from_zip(path.as_path(), "TreatSummary.csv").expect("could not find summary");
                    // Write the CSV contents to a temporary file
                    // Create a new directory with the same name as the zip file
                    let dir_path = path.with_extension("");
                    std::fs::create_dir_all(&dir_path).expect("Failed to create directory");
                    self.filepath = Some(dir_path.join("TreatSummary.csv"));
                    std::fs::write(self.filepath.as_ref().unwrap().as_path(), csv_contents).expect("No File");
                    
                    // Load the CSV file into the data frame
                    self.df = Some(read_csv_file(self.filepath.as_ref().unwrap().as_path()).expect("could not read"));
                    // Extract BMP images from zip files in the same directory
                    if self.extract_images {
                        extract_zip(path.as_path(), &dir_path);
                    }
                }
            };
            });
            ui.with_layout(egui::Layout::top_down_justified(egui::Align::Center), |ui| {
                ui.checkbox(&mut self.extract_images, RichText::new("Extract Images").size(25.0));
            });
            ui.with_layout(egui::Layout::top_down_justified(egui::Align::Center), |ui| {
                if ui.button(RichText::new("From Snapshots").size(25.0)).clicked() {
                    if let Some(path) = rfd::FileDialog::new().pick_folder() {
                        let dir_path = path.clone();
                        let new_dir_path = dir_path.join(format!("{}_extracted", dir_path.to_str().unwrap_or_default()));
                        if std::fs::create_dir_all(&new_dir_path).is_err() {
                            eprintln!("Failed to create directory");
                        } else {
                            match std::fs::read_dir(&dir_path) {
                                Ok(entries) => {
                                    for entry in entries {
                                        match entry {
                                            Ok(entry) => {
                                                if let Some(extension) = entry.path().extension() {
                                                    if extension == "zip" {
                                                        extract_zip(entry.path().as_path(), &new_dir_path);
                                                    }
                                                }
                                            },
                                            Err(_) => eprintln!("Failed to read entry"),
                                        }
                                    }
                                },
                                Err(_) => eprintln!("Failed to read directory"),
                            }
                        }
                    }
                };
            });

            if self.df.is_some() {
                let df = self.df.as_ref().unwrap();
                egui::ScrollArea::both().show(ui, |ui| {
                    Grid::new("dataframe_grid").show(ui, |ui| {
                        // Add column names to the grid
                        let column_names = df.get_column_names();
                        for name in &column_names {
                            ui.label(*name);
                        }
                        ui.end_row();

                        // Add values to the grid
                        for row_idx in 0..df.height() {
                            for name in &column_names {
                                let series = df.column(name).unwrap();
                                let value = series.get(row_idx).unwrap();
                                ui.label(format!("{:.2}", value));
                            }
                            ui.end_row();
                        }
                    });
                });
            }
        });
    }
    #[inline(always)]
    fn show_parameter_ui(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                let ejs = self.power as f64
                    * (self.pulseduration * 1e-3)
                    * self.pulsetrain as f64
                    * self.cycles as f64;
                let totduration = self.reptime * self.cycles as f64;
                let dutycycle =
                    ((self.pulsetrain as f64 * (self.pulseduration * 1e-3) * self.subspots as f64)
                        / self.reptime)
                        * 100.0;
                let recphase = ((self.reptime
                    - (self.pulsetrain as f64 * (self.pulseduration * 1e-3) * self.subspots as f64))
                    / (self.pulsetrain as f64 * self.subspots as f64))
                    * 1000.0;
                let period = (((self.reptime
                    - (self.pulsetrain as f64 * (self.pulseduration * 1e-3) * self.subspots as f64))
                    / (self.pulsetrain as f64 * self.subspots as f64))
                    * 1000.0)
                    + self.pulseduration;
                let dts =
                    ((self.pulsetrain as f64 * (self.pulseduration * 1e-3)) / self.reptime) * 100.0;
                let targetvol = self.spacing * 0.1 * self.spacing * 0.1 * 0.7 * self.subspots as f64;
                let epm = ejs / (self.spacing * self.spacing * 7.0);
                let prf = (self.subspots as f64 * self.pulsetrain as f64) / self.reptime;
                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        ui.spacing_mut().slider_width = 0.4*ui.available_width();
                        ui.add(
                            egui::Slider::new(&mut self.power, 0.0..=100.0)
                                .text(RichText::new("Power (W)").size(20.0))
                                .clamp_to_range(false),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.subspots, 1..=64)
                                .text(RichText::new("# of Subspots").size(20.0))
                                .clamp_to_range(false),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.spacing, 2.0..=4.0)
                                .text(RichText::new("Subspot Spacing").size(20.0))
                                .clamp_to_range(false),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.pulsetrain, 1..=10)
                                .text(RichText::new("Pulse Train").size(20.0))
                                .clamp_to_range(false),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.pulseduration, 2.4..=20.0)
                                .text(RichText::new("Pulse Duration (ms)").size(20.0))
                                .clamp_to_range(false),
                        );
                        ui.add(egui::Slider::new(&mut self.reptime, 0.00..=10.00)
                                    .text(RichText::new("Repetition Time (s)").size(20.0))
                                    .clamp_to_range(false));
                        ui.add(
                            egui::Slider::new(&mut self.cycles, 1..=200)
                                .text(RichText::new("# of Repetitions").size(20.0))
                                .clamp_to_range(false),
                        );
                    });
                    ui.vertical(|ui| {
                        let color = match (period < 4.0, dutycycle > 0.75, recphase < 1.6) {
                            (true, _, _) | (_, true, _) | (_, _, true) => Color32::RED,
                            _ => Color32::BLACK,
                        };
                        ui.label(RichText::new(format!("Duty Cycle {:.2} %", dutycycle)).size(20.0).color(color).underline());
                        ui.label(RichText::new(format!("Duty Cycle per Subspot {:.2} %", dts)).size(20.0).color(color).underline());
                        ui.label(RichText::new(format!("Energy per Subspot {:.2} %", ejs)).size(20.0).color(color).underline());
                        ui.label(RichText::new(format!("Pulse Repetition Frequency {:.2}", prf)).size(20.0).color(color).underline());
                        ui.label(RichText::new(format!("Period {:.2}", period)).size(20.0).color(color).underline());
                        ui.label(RichText::new(format!("Receiver Phase {:.2}", recphase)).size(20.0).color(color).underline());
                    });

                });
                let current_time = Local::now();
                let new_row = vec![
                                    format!("{}", self.sonication_number),
                                    format!("{}", current_time.format("%Y-%m-%d\n%H:%M:%S")),
                                    format!("{}", self.power),
                                    format!("{}", self.subspots),
                                    format!("{}", self.spacing),
                                    format!("{}", self.pulsetrain),
                                    format!("{:.2}", self.pulseduration),
                                    format!("{:.2}", self.reptime),
                                    format!("{:.2}", self.cycles),
                                    format!("{:.2}", ejs),
                                    format!("{:.2}", targetvol),
                                    format!("{:.1}", epm),
                                    format!("{:.1}", totduration),
                                    format!("{:.1}", prf),
                                    format!("{:.2}", recphase),
                                    format!("{:.2}", period),
                                    format!("{:.1}", dutycycle),
                                    format!("{:.1}", dts),
                                ];
                // Create a Grid widget to display the grid
                ui.with_layout(egui::Layout::top_down_justified(egui::Align::Center), |ui| {

                    egui::Grid::new("sonication_grid")
                        .spacing(egui::vec2(8.0, 10.0))
                        .show(ui, |ui| {
                        // Add data rows to the grid
                        for row in &self.grid_data {
                            for cell in row {
                                ui.label(cell);
                            }
                            ui.end_row();
                        }
                        // Add new_row to the grid
                        for cell in &new_row {
                            ui.label(cell);
                        }
                        ui.end_row();
                    });
                    if ui.button("Save").clicked() {
                        self.grid_data.push(new_row.clone());
                        self.sonication_number += 1;
                    };
                    ui.add(egui::TextEdit::singleline(&mut self.summaryname));
                    if ui.button("Export").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .set_file_name("Select a folder")
                            .pick_folder() {
                                let path =Path::new(&path);
                                let file = File::create(path.join(&self.summaryname)).expect("could not create file");
                                let mut wtr = Writer::from_writer(file);

                                for row in &self.grid_data {
                                    wtr.write_record(row).expect("csv write failed");
                                }

                                wtr.flush().expect("failed to close");
                            }
                    }
                });
            });
        });
    }

    #[inline(always)]
    fn show_dicom_ui(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.with_layout(egui::Layout::top_down_justified(egui::Align::Center), |ui| {

                if ui.button(RichText::new("Select folder")).clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .set_file_name("Select a folder")
                        .pick_folder() {
                        self.selected_folder = Some(path);
                    };


                    if let Some(ref folder) = self.selected_folder {
                        let path = Path::new(folder);

                        let mut objects: Vec<_> = WalkDirGeneric::<(u32, bool)>::new(path)
                            .process_read_dir(|_depth, _path, read_dir_state, _children| {
                                *read_dir_state += 1;
                            })
                            .into_iter()
                            .filter_map(Result::ok)
                            .filter(is_dicom)
                            .filter_map(to_dicom)
                            .collect();

                        objects.into_par_iter().for_each(|(object, file)| {
                            if let Ok(series_instance_uid) = object.element(Tag(0x0020, 0x000E)) {
                                if let Ok(uid) = series_instance_uid.to_str() {
                                    let mut series = self.presorted.entry(uid.to_string()).or_insert(Vec::new());
                                    series.push((object, file));
                                    series.par_sort_by(|(a, _), (b, _)| {
                                        let image_position_patient_a = get_image_position(a);
                                        let image_position_patient_b = get_image_position(b);
                                        image_position_patient_a
                                            .partial_cmp(&image_position_patient_b)
                                            .unwrap_or(Ordering::Equal)
                                    });
                                }
                            }
                        });
                    };
                };

                if let Some(ref folder) = self.selected_folder {
                    let path = Path::new(folder);
                    if ui.button("Move sorted dicoms").clicked() {

                        if let Some(objects) = self.presorted.get(&self.selected_uid) {
                            let new_dir = path.join(format!("processed/sorted/{}", &self.selected_uid));
                            std::fs::create_dir_all(&new_dir).expect("Failed to create directory");

                            for (object, file) in objects.value() {
                                let new_path = new_dir.join(file.file_name().expect("Failed to get file name"));
                                object.write_to_file(new_path).expect("Unable to write DICOM object");
                            }
                        } else {
                            // Process each object and its path in parallel using rayon
                            self.presorted.par_iter().for_each(|entry| {
                                let series = entry.key();
                                let objects = entry.value();
                                let new_dir = path.join(format!("processed/sorted/{}", series));
                                std::fs::create_dir_all(&new_dir).expect("Failed to create directory");

                                for (object, file) in objects {
                                    let new_path = new_dir.join(file.file_name().expect("Failed to get file name"));
                                    object.write_to_file(new_path).expect("Unable to write DICOM object");
                                }
                            });
                        }
                    };
                    if ui.button("Anonymize and save dicoms").clicked() {
                        // Anonymize each object in parallel using rayon
                        let tags_to_anonymize: HashSet<Tag> = [
                            Tag(0x0008, 0x0014), // Instance Creator UID
                            Tag(0x0008, 0x0018), // SOP Instance UID
                            Tag(0x0008, 0x0050), // Accession Number
                            Tag(0x0008, 0x0080), // Institution Name
                            Tag(0x0008, 0x0081), // Institution Address
                            Tag(0x0008, 0x0090), // Referring Physician's Name
                            Tag(0x0008, 0x0092), // Referring Physician's Address
                            Tag(0x0008, 0x0094), // Referring Physician's Telephone numbers
                            Tag(0x0008, 0x1010), // Station Name
                            Tag(0x0008, 0x1030), // Study Description
                            Tag(0x0008, 0x103E), // Series Description
                            Tag(0x0008, 0x1040), // Institutional Department name
                            Tag(0x0008, 0x1048), // Physician(s) of Record
                            Tag(0x0008, 0x1050), // Performing Physicians' Name
                            Tag(0x0008, 0x1060), // Name of Physician(s) Reading study
                            Tag(0x0008, 0x1070), // Operator's Name
                            Tag(0x0008, 0x1080), // Admitting Diagnoses Description
                            Tag(0x0008, 0x1155), // Referenced SOP Instance UID
                            Tag(0x0008, 0x2111), // Derivation Description
                            Tag(0x0010, 0x0010), // Patient's Name
                            Tag(0x0010, 0x0020), // Patient ID
                            Tag(0x0010, 0x0030), // Patient Birth Day
                            Tag(0x0010, 0x0032), // Patient's Birth Time
                            Tag(0x0010, 0x0040), // Patient's Sex
                            Tag(0x0010, 0x1010), // Patient's Age
                            Tag(0x0014, 1001),   // Custom Tag
                        ]
                        .iter()
                        .cloned()
                        .collect();
                        self.presorted.par_iter_mut().for_each(|mut entry| {
                            let series = entry.key().clone();
                            let objects = entry.value_mut();
                            let new_dir = path.join(format!("processed/anonymized/{}", series));
                            std::fs::create_dir_all(&new_dir).expect("Failed to create directory");

                            for (object, file) in objects {
                                // List of tags to anonymize
                                for tag in &tags_to_anonymize {
                                    object.remove_element(*tag);
                                }
                                // Save the anonymized DICOM object back to disk with a new name
                                let new_file_name = format!(
                                    "{}_anonymized.dcm",
                                    file.file_stem().unwrap().to_str().unwrap()
                                );
                                let new_path = new_dir.join(new_file_name);
                                object.write_to_file(new_path).expect("Unable to write DICOM object");
                            }
                        });
                    }
                };
            });
                // Print the series and number of images
                if let Some(ref _folder) = self.selected_folder {

                    egui::ScrollArea::vertical().show(ui, |ui| {
                        egui::SidePanel::right("side_panel").show(ui.ctx(), |ui| {
                            ui.label(RichText::new("Series and number of images:").size(25.0));
                            let mut selected_series = None;
                            egui::ComboBox::from_id_source("my_combo_box")
                                .selected_text(self.selected_uid.clone())
                                .width(ui.available_width()) // Set the width here
                                .show_ui(ui, |ui| {
                                    for entry in &self.presorted {
                                        let series = entry.key();
                                        let count = entry.value().len();
                                        ui.selectable_value(&mut selected_series, Some(series.to_string()), format!("{}: {}", series, count));
                                    }
                                });
                            if let Some(selected) = selected_series {
                                self.selected_uid = selected;
                                self.current_image_index = 1;
                            };
                            if let Some(images) = self.presorted.get(&self.selected_uid) {
                                ui.horizontal(|ui| {

                                    let (dicom_object, path) = &images[self.current_image_index - 1];
                                    // Display the image from dicom_object
                                    let pixel_data = &dicom_object.decode_pixel_data().unwrap();
                                    let size = [pixel_data.rows() as _, pixel_data.columns() as _];
                                    let dynamic_image = pixel_data.to_dynamic_image(0).unwrap().to_rgba8();
                                    let pixels = dynamic_image.as_flat_samples();
                                    let image = ColorImage::from_rgba_unmultiplied(size,pixels.as_slice());
                                    let texture_options = egui::TextureOptions::default(); // or any other options you want to set
                                    let texture: &egui::TextureHandle = &ui.ctx().load_texture("0", image, texture_options);
                                    ui.image(texture);
                                    ui.spacing_mut().slider_width = ui.available_height();
                                    ui.add(egui::Slider::new(&mut self.current_image_index, 1..=images.len())
                                        .orientation(SliderOrientation::Vertical)
                                        .text("Image index"),
                                    );
                                });

                            };
                        });
                    });
                }
        });
    }
}

impl eframe::App for MyEguiApp {
    #[inline(always)]
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("my_top_panel").show(ctx, |ui| {
            menu::bar(ui, |ui| {
                ui.horizontal(|ui| {
                    
                    ui.menu_button(RichText::new("View").size(15.0), |ui| {
                        egui::gui_zoom::zoom_menu_buttons(ui, frame.info().native_pixels_per_point);
                    });
                    ui.separator();
                    ui.menu_button(RichText::new("Contrast").size(15.0), |ui| {
                        egui::widgets::global_dark_light_mode_buttons(ui);
                    });
                    ui.separator();
                    ui.menu_button(RichText::new("Mode").size(15.0), |ui| {
                        ui.vertical(|ui| {
                            if ui.button("Parameter Calculator").clicked() {
                                self.show_mode = "parameters".into()
                            };
                            if ui.button("Treatment Summary").clicked() {
                                self.show_mode = "summary".into()
                            };
                            if ui.button("dicom tools").clicked() {
                                self.show_mode = "dicom".into()
                            };
                        });
                    });
                });
            });
        });
        match &*self.show_mode {
            "parameters" => self.show_parameter_ui(ctx, frame),
            "summary" => self.show_summary_ui(ctx, frame),
            "dicom" => self.show_dicom_ui(ctx, frame),
            _ => (), // handle other cases
        }
    }
}
