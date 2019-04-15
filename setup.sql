
CREATE TABLE IF NOT EXISTS people (
    nickname text PRIMARY KEY,
    /* TODO full name too? */
    /* maybe normalize these somewhere if going to use this field for automated
     * emails, texts, etc */
    email text,
    mobile_phone text,
    alt_phone text
    /*
    default_num_seed_males smallint CHECK (default_num_seed_males > 0),
    default_num_seed_females smallint CHECK (default_num_seed_females > 0),
    default_add_yeast boolean
    */
    /* TODO provide defaults for other values we will want input on. maybe
     * protocols? make their handling generic? */
);

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

CREATE TABLE IF NOT EXISTS code_versions (
    /* TODO do i want this to be the PK? it will surely lead to duplicate rows
     * referring to the same code version, but maybe it's not worth checking to
     * prevent duplicates, considering maybe small space savings? */
    version_id SERIAL PRIMARY KEY,

    /* Could be a Python package/module name. */
    name text NOT NULL,

    /* For when the code is used directly from source. */
    git_remote text,
    git_hash text,
    git_uncommitted_changes text,

    /* Any string version number, but with Python code, this should be the
     * output of pkg_resources.get_distribution(<pkg_name>).version
     * This should generally agree with import package; package.__version__
     * To be used when the installed version is not being used from source. */
    version text,

    CONSTRAINT all_git_info CHECK (git_hash is null or
        (git_remote is not null and git_uncommitted_changes is not null)),
    CONSTRAINT have_version CHECK (git_hash is not null or version is not null)
);

/* TODO TODO TODO make sure these is enough references between this and other
 * tables, so that output generated from a certain cnmf run can be deleted, and
 * at least expose this in GUI (so people can retroactively select and reject
 * stuff) */
/* TODO rename to something more general, like trace_extraction_runs? */
CREATE TABLE IF NOT EXISTS analysis_runs (
    run_at timestamp PRIMARY KEY,

    /* TODO some check to try to ensure these don't accidentally refer to the
     * same thing? */

    /* TODO TODO support tracking other repos besides one driver repo?
       like since i call some of Remy's matlab code from it to extract timing
       information... */
    /* TODO option in GUI to add arbitrary repos to track? */
    /* TODO maybe store driver version is some other table, alongside a ref to
     * the cnmf_run? something like analysis_runs? but then should that also
     * include motion correction and stuff? i don't currently have stored
     * version info for all of those corrected movies... */

    /* This is code that calls the CaImAn library code. */
    /*
    driver_version integer REFERENCES code_versions (version_id) NOT NULL,
    */
    /* TODO is this really the only thing that will ultimately be unique to
     * CNMF? remove not null constraint then? */
    /* This is the CaImAn library, which you may not want to change. */
    /*caiman_version integer REFERENCES code_versions (version_id) NOT NULL,
    other_versions 
    */

    /* TODO store all manual input along the way (when / what for component
     * deletions, creation, decisions on manual control of iteration #, param
       changes along the way, etc) */

    input_filename text NOT NULL,
    input_md5 text NOT NULL,
    input_mtime timestamp NOT NULL,
    /* ctime worth it? */

    /* Use NULL to indicate from start / to end */
    start_frame integer,
    stop_frame integer,

    /* TODO TODO also store references to date/fly/recording stuff?
       required or optional? */

    /* Could use json/jsonb type, but same values, like Infinity, would have to
     * be handled differently. */
    parameters text NOT NULL,

    who text REFERENCES people (nickname),
    /* Partially to figure out who ran it if they forgot to select their name.
    */
    /* TODO also check these are at least non-empty */
    host text NOT NULL,
    host_user text NOT NULL,

    accepted boolean NOT NULL

    /*  TODO TODO TODO add constraint to this effect:
     * probably want to ensure all git stuff + input + parameters + user
     * uniqueness... but maybe not w/ pk? */
    /* TODO though may need to be careful if user is part of pk, just so people
       don't accidentally work w/ the wrong version of traces (including >1
       version) */
);


/* This table will N rows for each analysis run, where N is the number of peices
 * of software whose versions we are tracking. */
CREATE TABLE IF NOT EXISTS analysis_code (
    run_at timestamp REFERENCES analysis_runs (run_at),
    version_id integer REFERENCES code_versions (version_id),
    PRIMARY KEY(run_at, version_id)
);


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
/* TODO "if not exists" or something, to prevent error? */
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
