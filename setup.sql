
CREATE TABLE IF NOT EXISTS flies (
    /* TODO serial? work w/ pandas insert if just not specified? */
    /*fly smallserial UNIQUE NOT NULL, */

    prep_date date NOT NULL,
    /* The fly's order within the day. */
    fly_num smallint NOT NULL,

    /* TODO also load fly_num column from google sheet and use that w/ prep_date
     * to generate fly PK? */
    /* TODO appropriate constraints on each of these? */
    indicator text,
    days_old smallint,
    surgeon text,
    surgery_quality smallint,
    notes text,

    PRIMARY KEY(prep_date, fly_num)
);

CREATE TABLE IF NOT EXISTS odors (
    /* TODO maybe use CAS or some other ID instead? */
    name text NOT NULL,
    log10_conc_vv real NOT NULL,

    odor smallserial UNIQUE NOT NULL,

    PRIMARY KEY(name, log10_conc_vv)
);

/* TODO or just make fk based pk w/ all fields of each odor duplicated? */
CREATE TABLE IF NOT EXISTS mixtures (
    /* TODO or just use name + conc? */
    odor1 smallint REFERENCES odors (odor) NOT NULL,
    odor2 smallint REFERENCES odors (odor) NOT NULL,
    PRIMARY KEY(odor1, odor2)
);

/* TODO table for stimulus info? */
/* TODO maybe stimulus code or version somewhere if nothing else? */

/* TODO maybe get rid of this table? if only going to be used 1-1 w/ responses?
   oh, but footprints too? that enough reason?
 * */
CREATE TABLE IF NOT EXISTS analysis_runs (
    analysis_description text NOT NULL,
    /* TODO precision? i think at least seconds? otherwise whatever is
     * convenient to work from time.time() in python... */
    analyzed_at timestamp NOT NULL,
    /* TODO actually any simpler making some kind of artificial key like this
     * for fk purposes? i didn't want to have to refer to two extra columns in
     * responses keys... */
    analysis_run smallserial UNIQUE NOT NULL,

    /* TODO git version + unsaved changes? */
    /* TODO git remotes? */
    PRIMARY KEY(analysis_description)
);

/* TODO delete analysis_runs? */
CREATE TABLE IF NOT EXISTS cnmf_runs (
    run_at timestamp PRIMARY KEY,

    /* TODO TODO check that either these or pkg version are supplied */
    git_remote text,
    git_hash text,
    git_uncommitted_changes text,

    version text,

    input_filename text NOT NULL,
    input_md5 text NOT NULL,
    /* TODO correct dtype? mtime what i want? */
    input_mtime timestamp NOT NULL,

    /* Use NULL to indicate from start / to end */
    start_frame integer,
    stop_frame integer,

    /* TODO TODO also store references to date/fly/recording stuff?
       required or optional? */

    /* Could use json/jsonb type, but same values, like Infinity, would have to
     * be handled differently. */
    parameters text NOT NULL,

    accepted boolean NOT NULL,

    /* TODO TODO what to use as pk? just run_at? serial? */
    /* TODO probably want to ensure all git stuff + input + parameters
     * uniqueness... but maybe not w/ pk? */

    CONSTRAINT all_git_info CHECK (git_hash is null or
        (git_remote is not null and git_uncommitted_changes is not null)),
    CONSTRAINT have_version CHECK (git_hash is not null or version is not null)
);

/* TODO TODO TODO new table for each algorithm we want to tune parameters for,
 * with a column for each relevant parameter */
/* TODO TODO TODO and a new table for each function to optimize, storing
 * references to the other tables (how to implement 1->many tables?) and
 * performance on the function */

CREATE TABLE IF NOT EXISTS recordings (
    /*recording_num smallserial PRIMARY KEY,*/
    /* TODO appropriate precision? */
    started_at timestamp PRIMARY KEY,
    /* TODO check thorsync and thorimage are unique (w/in day at least?)
       require longer path and just check unique across all?
    */

    /* TODO just use combination of paths as primary key */
    thorsync_path text NOT NULL,
    /* TODO check this is not null in the responses case? */
    /* This is nullable so that recordings can also be used for PID recordings.
     * */
    thorimage_path text,
    /* TODO maybe require this if it's just going to be the pin/odor info? */
    stimulus_data_path text

    /* TODO store framerate here? so analysis on movies can do stuff for fixed
     * amount of seconds w/o having to load xml? */
);

/* TODO table for frame indices used to define responses? store alongside
 * responses, particularly if from_onset and df_over_f can be make into array
 * types? make a separate presentations / blocks table again? */

/* TODO TODO just combine this w/ responses table (maybe frame #s are slightly
 * inconsistent w/ bounds on timeseries in responses?) */
CREATE TABLE IF NOT EXISTS presentations (
    /* TODO maybe remove these parts of primary key? */
    prep_date timestamp NOT NULL,
    fly_num smallint NOT NULL,

    /* TODO TODO maybe some kind of alternate index w/ thorimage_id as opposed
     * to recording from, derived in postgres? */
    recording_from timestamp REFERENCES recordings (started_at) NOT NULL,

    /* TODO maybe reference an odor pair here? */
    comparison smallint NOT NULL,

    odor1 smallint NOT NULL,
    odor2 smallint NOT NULL,

    repeat_num smallint NOT NULL,
    
    /* TODO make include other key frames too? or relegate to block info? */
    odor_onset_frame integer NOT NULL,
    odor_offset_frame integer NOT NULL,

    /* Timing information of same dimensions as corresponding cell entries in
     * responses table. */
    from_onset double precision[] NOT NULL,

    /* TODO want this nullable or not? */
    analysis smallint REFERENCES analysis_runs (analysis_run),

    presentation_id SERIAL UNIQUE NOT NULL,

    /* TODO TODO label as to whether comparison had good quality data,
    perhaps separately judged on 1) motion and 2) consistency (correlation) w/in
    any repeats */

    FOREIGN KEY(prep_date, fly_num) REFERENCES flies(prep_date, fly_num),
    FOREIGN KEY(odor1, odor2) REFERENCES mixtures(odor1, odor2),
    PRIMARY KEY(prep_date, fly_num, recording_from,
                comparison, odor1, odor2, repeat_num)
);
ALTER TABLE presentations
    ADD CONSTRAINT from_onset_len
    CHECK (cardinality(from_onset) > 1);

/* TODO worth having a separate representation of cells that is indep. calls on
 * a frame / block basis? (which can be associated together to get this)? */
/* TODO or store in this table too somehow? */
/* TODO TODO store ground truth contour / mask here? elsewhere? (including
 * manually created cells?) */
CREATE TABLE IF NOT EXISTS cells (
    /* TODO make pk more consistent w/ presentations? get rid of fly / prep_date
     * there? add it here? */
    recording_from timestamp REFERENCES recordings (started_at) NOT NULL,
    cell smallint NOT NULL,

    /* TODO constraint to check these are all same length? just define a new
     * type? (though that might be harder to insert into...) */

    x_coords smallint[] NOT NULL,
    y_coords smallint[] NOT NULL,
    /* TODO appropriate precision here? */
    weights real[] NOT NULL,

    /* For manually labelling two important types of segmentation errors. */
    /* TODO maybe change to a one character code describing ROI characteristics?
     */
    /* TODO TODO TODO move to comparison level!! (maybe put in presentations and
     * then just label each w/in a comparison the same? otherwise need new
     * table?) */
    only_one_cell boolean,
    all_of_cell boolean,
    /* TODO maybe allow drawing moving contour / mask for richer ground truth
     * here? */
    /* TODO TODO TODO make at least this flag (or all?) assignable for each
    (cell, comparison) not just each cell across all comparisons in recording */
    /* TODO or maybe each trial? arbitrary intervals? */
    stable boolean,
    /* TODO don't use boolean so you can label something as explicitly unsure?
       (diff from default NULL, which would just be "unlabeled") */

    /* TODO want this nullable or not? */
    analysis smallint REFERENCES analysis_runs (analysis_run),

    /* TODO maybe one wrt average and one wrt activity over time? */
    /* TODO per presentation? */
    /*good bool*/

    PRIMARY KEY(recording_from, cell)
);

/* TODO TODO store automated response calls in this table as well? */
/* TODO if so, how to store algorithm / version / parameters? multiple diff
 * calls? */
CREATE TABLE IF NOT EXISTS responses (
    /* TODO matter whether fk is an ID vs all columns of composite pk in other
     * table, as far as space / speed performance? */
    presentation_id integer REFERENCES presentations (presentation_id) NOT NULL,

    /* Redundant w/ information in presentation_id, but seems unavoidable in
     * order to include FK on cell... */
    recording_from timestamp,

    /* TODO could reference a per trial repr of cell boundaries here, if i'm
     * going to store such a representation... */
    cell smallint,

    df_over_f real[] NOT NULL,
    raw_f real[] NOT NULL,

    /* TODO TODO some nullable label of whether response should be considered a
     * response looking just at trace. */
    response boolean,

    /*
       TODO and maybe something else saying whether response came from only
       this footprint or not?
    */

    FOREIGN KEY(recording_from, cell) REFERENCES cells (recording_from, cell),
    PRIMARY KEY(presentation_id, recording_from, cell)
);
ALTER TABLE responses
    ADD CONSTRAINT raw_f_len
    CHECK (cardinality(raw_f) > 1);

ALTER TABLE responses
    ADD CONSTRAINT df_over_f_len
    CHECK (cardinality(df_over_f) > 1);

/* Would probably speed up inserts, but might be too risky. If db crashes, table
 * contents will probably be deleted. */
/* ALTER TABLE responses SET UNLOGGED; */

CREATE TABLE IF NOT EXISTS pid (
    /*
    mixture smallint REFERENCES mixtures (mixture) NOT NULL,
    */
    odor1 smallint,
    odor2 smallint,

    recording_from timestamp REFERENCES recordings (started_at) NOT NULL,
    /* TODO positive / nonneg constraint. alt repr? */
    repeat_num smallint NOT NULL,

    from_onset double precision[] NOT NULL,
    pid_out real[] NOT NULL,

    FOREIGN KEY(odor1, odor2) REFERENCES mixtures(odor1, odor2),
    PRIMARY KEY(odor1, odor2, recording_from, repeat_num)
);
ALTER TABLE pid
    ADD CONSTRAINT from_onset_len
    CHECK (cardinality(from_onset) > 1);

ALTER TABLE pid
    ADD CONSTRAINT pid_out_len
    CHECK (cardinality(pid_out) > 1);

/* TODO link to table w/ PID / flow settings or just include for each row here?
 * */

/* TODO worth keeping a representation of traces across whole experiment?
   convert everything to odor responses?

   just include start frame / time for each chunk?
 */

/* TODO store footprints? how to represent? */
