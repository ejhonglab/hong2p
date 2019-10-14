
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
    prep_date date NOT NULL,
    -- The fly's order within the day.
    fly_num smallint NOT NULL,

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
    /* TODO could assert this is non-positive */
    log10_conc_vv real NOT NULL,

    odor_id smallserial UNIQUE NOT NULL,

    PRIMARY KEY(name, log10_conc_vv)
);


/* TODO or just make fk based pk w/ all fields of each odor duplicated? */
/* TODO TODO refactor to many-to-many between odors and mixtures, where mixtures
 * are either just given serial IDs, or also checked for uniqueness across group
 * of odors. how is that normally implemented? */
CREATE TABLE IF NOT EXISTS mixtures (
    /* TODO or just use name + conc? */
    odor1 smallint REFERENCES odors (odor_id) NOT NULL,
    odor2 smallint REFERENCES odors (odor_id) NOT NULL,
    PRIMARY KEY(odor1, odor2)
);


/* TODO TODO TODO want to enforce that only one mixture has a specific 
   (order agnostic) combination of odors */
/* TODO maybe have a trigger that computes hash unique to set of odors,
   over all odors of recently inserted/generated mixture, and then enforce
   uniqueness on that hash? more SQL / postgres idiomatic way to accomplish
   this?*/

/* TODO see:
https://dba.stackexchange.com/questions/235291/unique-sets-of-ids-from-table
for a possible solution */

/* TODO fix implementation and use this rather than above mixture table
CREATE TABLE IF NOT EXISTS mixtures (
    mixture_id smallserial PRIMARY KEY,
);


CREATE TABLE IF NOT EXISTS mixture_odors (
    mixture_id smallint REFERENCES mixtures (mixture_id),
    odor_id smallint REFERENCES odors (odor_id)
    -- TODO maybe also worth making these unique?
);
*/


CREATE TABLE IF NOT EXISTS recordings (
    /* rename to recording_time? */
    started_at timestamp PRIMARY KEY,
    /* TODO check thorsync and thorimage are unique (w/in day at least?)
       require longer path and just check unique across all?
    */

    /* TODO use combination of paths as primary key? */
    thorsync_path text NOT NULL,
    /* TODO check this is not null in the responses case? */
    /* This is nullable so that recordings can also be used for PID recordings.
     * */
    thorimage_path text,
    /* TODO maybe require this if it's just going to be the pin/odor info? */
    stimulus_data_path text,
    /* TODO stimulus code here too? (opt if also aiming to support PID-only)*/

    full_frame_avg_trace real[],

    -- Should match first_block / last_block derived from gsheet
    first_block smallint,
    last_block smallint,
    n_repeats smallint,
    presentations_per_repeat smallint
);
ALTER TABLE recordings
    ADD COLUMN full_frame_avg_trace real[];
-- TODO maybe make block info NOT NULL, since depend on it now in
-- util.accepted_blocks...
ALTER TABLE recordings
    ADD COLUMN first_block smallint;
ALTER TABLE recordings
    ADD COLUMN last_block smallint;
ALTER TABLE recordings
    ADD COLUMN n_repeats smallint;
ALTER TABLE recordings
    ADD COLUMN presentations_per_repeat smallint;


CREATE TABLE IF NOT EXISTS code_versions (
    /* TODO do i want this to be the PK? it will surely lead to duplicate rows
     * referring to the same code version, but maybe it's not worth checking to
     * prevent duplicates, considering maybe small space savings? */
    version_id SERIAL PRIMARY KEY,

    /* Could be a Python package/module name. */
    name text NOT NULL,
    /* To disambiguate two versions of the same package associated with one
     * analysis run. */
    used_for text,

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


CREATE TABLE IF NOT EXISTS analysis_runs (
    /* rename to analysis_time? */

    -- If entry describes analysis using ImageJ ROIs, run_at will be the mtime
    -- of the ROI .zip file.
    -- TODO might want to use a new field for ijroi file mtime and unique on
    -- that as well... so that the code can be updated?
    -- maybe ok to replace old versions then though?
    run_at timestamp PRIMARY KEY,

    /* TODO option in GUI to add arbitrary repos to track?
       see analysis_code table. */
    recording_from timestamp REFERENCES recordings (started_at) NOT NULL,
    /* TODO TODO maybe only add these to a table that references analysis runs,
       like maybe human_checked_analysis_runs?

       everything from input_* to accepted are probably not necessary in some of
       the uses for this table. i mostly just want to group the analysis_code
       rows in the same way in those cases.

       (considering presentations requirement for analysis and lack of any check
       on that portion of the analysis...)
       could also either just set this to true / make nullable and leave null
    */
    -- TODO TODO only check NOT NULL constraint from segmentation_runs table?
    -- (seems impossible. would probably require diff design.)
    input_filename text,
    input_md5 text,
    input_mtime timestamp,

    -- Use NULL to indicate from start / to end
    start_frame integer,
    stop_frame integer,

    -- Could use json/jsonb type, but same values, like Infinity, would have to
    -- be handled differently.
    parameters text,

    ijroi_file_path text,

    who text REFERENCES people (nickname),
    -- Partially to figure out who ran it if they forgot to select their name.
    -- TODO also check these are at least non-empty
    host text,
    host_user text,

    accepted boolean

    /* TODO TODO enforce uniqueness of all git stuff + input + parameters
     * (+ user)?

       if keeping user... no real need to recompute if only user differs, right?
       (assuming output was accepted / saved last time)
     */
);
ALTER TABLE analysis_runs ALTER COLUMN input_filename DROP NOT NULL;
ALTER TABLE analysis_runs ALTER COLUMN input_md5 DROP NOT NULL;
ALTER TABLE analysis_runs ALTER COLUMN input_mtime DROP NOT NULL;
ALTER TABLE analysis_runs ALTER COLUMN parameters DROP NOT NULL;
ALTER TABLE analysis_runs ALTER COLUMN host DROP NOT NULL;
ALTER TABLE analysis_runs ALTER COLUMN host_user DROP NOT NULL;
ALTER TABLE analysis_runs ALTER COLUMN accepted DROP NOT NULL;
ALTER TABLE analysis_runs ADD COLUMN ijroi_file_path text;

/*
CREATE TABLE IF NOT EXISTS analysis_run_inputs (
    run_at timestamp REFERENCES analysis_runs (run_at),

    -- TODO TODO only check NOT NULL constraint from segmentation_runs table?
    input_filename text NOT NULL,
    input_md5 text NOT NULL,
    input_mtime timestamp NOT NULL,

    -- Use NULL to indicate from start / to end
    start_frame integer,
    stop_frame integer,

    PRIMARY KEY(run_at)
);


CREATE TABLE IF NOT EXISTS human_analysis_runs (
    run_at timestamp REFERENCES analysis_runs (run_at),

    -- TODO only want this here? make a table just for params? just add to each
    -- terminal table that needs them (e.g. segmentation_runs)?
    -- Could use json/jsonb type, but same values, like Infinity, would have to
    -- be handled differently.
    parameters text NOT NULL,

    who text REFERENCES people (nickname),
    -- Partially to figure out who ran it if they forgot to select their name.
    -- TODO also check these are at least non-empty
    host text NOT NULL,
    host_user text NOT NULL,

    accepted boolean NOT NULL,

    PRIMARY KEY(run_at)
);
*/


/* This table will N rows for each analysis run, where N is the number of pieces
 * of software whose versions we are tracking. */
CREATE TABLE IF NOT EXISTS analysis_code (
    run_at timestamp REFERENCES analysis_runs (run_at) ON DELETE CASCADE,
    -- TODO TODO how to get cascade deletes from above to case deletions of
    -- code_versions.version_id rows that this references?
    -- TODO or maybe more accurately, how to delete code_versions rows when
    -- they are no longer referenced? just need separate sql statements?
    version_id integer REFERENCES code_versions (version_id),
    PRIMARY KEY(run_at, version_id)
);
ALTER TABLE analysis_code
DROP CONSTRAINT analysis_code_run_at_fkey,
ADD CONSTRAINT analysis_code_run_at_fkey
    FOREIGN KEY (run_at)
    REFERENCES analysis_runs (run_at)
    ON DELETE CASCADE;


CREATE TABLE IF NOT EXISTS segmentation_runs (
    run_at timestamp REFERENCES analysis_runs (run_at) ON DELETE CASCADE,

    output_fig_png bytea NOT NULL,
    output_fig_svg bytea NOT NULL,
    output_fig_mpl bytea NOT NULL,

    -- TODO store ref (directly or indirectly) to analysis params for motion
    -- correction that produced this input? (keeping in mind curr plan is still
    -- to use analysis_runs table for morr corr as well)

    -- TODO store all manual input along the way (when / what for component
    -- deletions, creation, decisions on manual control of iteration #, param
    -- changes along the way, etc) (if i ever support manual input...)

    PRIMARY KEY(run_at)
);
-- TODO maybe move back to analysis_runs table?
ALTER TABLE segmentation_runs ADD COLUMN run_len_seconds real;
-- TODO enforce that there is at least one row in analysis_code referring to
-- each run?
ALTER TABLE segmentation_runs
DROP CONSTRAINT segmentation_runs_run_at_fkey,
ADD CONSTRAINT segmentation_runs_run_at_fkey
    FOREIGN KEY (run_at)
    REFERENCES analysis_runs (run_at)
    ON DELETE CASCADE;


/* TODO TODO TODO but how am i going to indicate which recordings / analysis
 * output has passed all other (possibly including manual) rejection criteria?
 */
/* TODO TODO should it be a flag comparison by comparison? just in presentations
 * table then? */
/* TODO for mocorr too? just refer to some analysis which also
 * refers back to mocorr? */
ALTER TABLE recordings
    ADD COLUMN canonical_segmentation timestamp
    REFERENCES segmentation_runs (run_at) ON DELETE SET NULL;

ALTER TABLE recordings
DROP CONSTRAINT recordings_canonical_segmentation_fkey,
ADD CONSTRAINT recordings_canonical_segmentation_fkey
    FOREIGN KEY (canonical_segmentation)
    REFERENCES segmentation_runs (run_at)
    ON DELETE SET NULL;


/* TODO worth having a separate representation of cells that is indep. calls on
 * a frame / block basis? (which can be associated together to get this)? */
/* TODO or store in this table too somehow? */
/* TODO TODO store ground truth contour / mask here? elsewhere? (including
 * manually created cells?) */
CREATE TABLE IF NOT EXISTS cells (
    /* TODO make pk more consistent w/ presentations? get rid of fly / prep_date
     * there? add it here? */
    /* TODO TODO maybe just refer to segmentation_run, and then get which input
     * data (and thus recording_from if really necessary) from there? */
    recording_from timestamp REFERENCES recordings (started_at) NOT NULL,
    /* TODO TODO TODO make sure however i give K data does not allow same data
     * w/ multiple segmentation_run
     
       maybe make some SQL constraint to indicate that each recording should
       only have one accepted analysis version? it would seem to limit my
       softwares usefulness for parameter exploration though, if i were to
       strictly prevent accepting output of multiple seg runs on same data...
     */
    segmentation_run timestamp NOT NULL REFERENCES segmentation_runs (run_at)
        ON DELETE CASCADE,
    cell smallint NOT NULL,

    /* TODO constraint to check these are all same length? just define a new
     * type? (though that might be harder to insert into...) */

    /* TODO TODO check these are at least length 1 like in traces? */
    x_coords smallint[] NOT NULL,
    y_coords smallint[] NOT NULL,
    /* TODO appropriate precision here? */
    weights real[] NOT NULL,

    /* For manually labelling two important types of segmentation errors. */
    /* TODO maybe change to a one character code describing ROI characteristics?
     */
    /* TODO TODO move to comparison level!! (maybe put in presentations and
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

    /* TODO maybe one wrt average and one wrt activity over time? */
    /* TODO per presentation? */

    PRIMARY KEY(recording_from, segmentation_run, cell)
);
ALTER TABLE cells
DROP CONSTRAINT cells_segmentation_run_fkey,
ADD CONSTRAINT cells_segmentation_run_fkey
    FOREIGN KEY (segmentation_run)
    REFERENCES segmentation_runs (run_at)
    ON DELETE CASCADE;


CREATE TABLE IF NOT EXISTS presentations (
    /* TODO maybe remove these parts of primary key? */
    prep_date timestamp NOT NULL,
    fly_num smallint NOT NULL,

    /* TODO TODO maybe some kind of alternate index w/ thorimage_id as opposed
     * to recording from, derived in postgres? */
    recording_from timestamp REFERENCES recordings (started_at) NOT NULL, 

    -- TODO maybe take analysis out of here... how important is it really?
    -- the chances of getting things like the framenumbers wrong, in a way
    -- that really matters, are probably pretty low?
    -- and relatively cheap to recalculate. exponential fitting also not super
    -- critical, right?
    -- I guess a reason to keep it in is that some relatively insignificant
    -- change might still kinda break depedent CNMF output, which we might not
    -- want.
    analysis timestamp NOT NULL REFERENCES analysis_runs (run_at)
        ON DELETE CASCADE,

    /* TODO maybe reference an odor pair here? */
    comparison smallint NOT NULL,

    odor1 smallint NOT NULL,
    odor2 smallint NOT NULL,

    repeat_num smallint NOT NULL,
    
    odor_onset_frame integer NOT NULL,
    odor_offset_frame integer NOT NULL,

    -- TODO maybe only check this exists if presentation_accepted is True?
    -- constraint like that possible?
    -- Timing information of same dimensions as corresponding cell entries in
    -- responses table.
    from_onset double precision[] NOT NULL,

    presentation_accepted boolean,

    /* TODO maybe store reference to just driver here then? and then
       either just segmentation or seg + driver in responses?
    */
    /* TODO TODO maybe refer to recordings (or the unit actually used as input
     * to analysis) in analysis_runs, and just use that as the linkage? */

    presentation_id SERIAL UNIQUE NOT NULL,

    /* TODO TODO label as to whether comparison had good quality data,
    perhaps separately judged on 1) motion and 2) consistency (correlation) w/in
    any repeats */

    FOREIGN KEY(prep_date, fly_num) REFERENCES flies(prep_date, fly_num),
    FOREIGN KEY(odor1, odor2) REFERENCES mixtures(odor1, odor2),
    PRIMARY KEY(prep_date, fly_num, recording_from, analysis,
                comparison, odor1, odor2, repeat_num)
);
ALTER TABLE presentations
    ADD CONSTRAINT from_onset_len
    CHECK (cardinality(from_onset) > 1);

/* Seconds from odor onset. t=0 to this time will not be used to calculate the
 * exponential decay. This should be roughly when the peak of the signal is. */
ALTER TABLE presentations
    ADD COLUMN calc_exp_from real;
ALTER TABLE presentations
    ADD COLUMN exp_scale real;
ALTER TABLE presentations
    ADD COLUMN exp_tau real;
ALTER TABLE presentations
    ADD COLUMN exp_offset real;
-- TODO keep these?
ALTER TABLE presentations
    ADD COLUMN exp_scale_sigma real;
ALTER TABLE presentations
    ADD COLUMN exp_tau_sigma real;
ALTER TABLE presentations
    ADD COLUMN exp_offset_sigma real;
-- TODO keep this one?
-- not exactly sure how to get things into right units here, if rescaling stuff
-- for numerical reasons. should be pretty easy to work out...
--ALTER TABLE presentations
--    ADD COLUMN exp_pcov real[];

ALTER TABLE presentations
    ADD COLUMN avg_dff_5s real;
ALTER TABLE presentations
    ADD COLUMN avg_zchange_5s real;
-- TODO add peak (amplitude) and calculated peak time

ALTER TABLE presentations 
DROP CONSTRAINT presentations_analysis_fkey,
ADD CONSTRAINT presentations_analysis_fkey
    FOREIGN KEY (analysis)
    REFERENCES analysis_runs (run_at)
    ON DELETE CASCADE;

ALTER TABLE presentations
    ADD COLUMN presentation_accepted boolean;


/* TODO TODO store automated response calls in this table as well? */
/* TODO if so, how to store algorithm / version / parameters? multiple diff
 * calls? */
CREATE TABLE IF NOT EXISTS responses (
    /* TODO matter whether fk is an ID vs all columns of composite pk in other
     * table, as far as space / speed performance? */
    presentation_id integer NOT NULL REFERENCES presentations (presentation_id)
        ON DELETE CASCADE,

    /* TODO maybe now use ID for cell rather than this triple? */
    /* Redundant w/ information in presentation_id, but seems unavoidable in
     * order to include FK on cell... */
    recording_from timestamp,
    segmentation_run timestamp,
    -- TODO do i need to specify ON DELETE CASCADE for this as well?
    -- (to avoid an error? if so, how to do that for part of composite fk?)
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

    /* TODO TODO TODO somehow only store responses for one canonical set of
     * analysis output, even if support accepting multiple footprints.
       i don't want to have to include segmentation_run in handling of this
       table. not sure i would actually use it...
     */
    FOREIGN KEY(recording_from, segmentation_run, cell)
        REFERENCES cells (recording_from, segmentation_run, cell),

    PRIMARY KEY(presentation_id, recording_from, cell)
);
/* TODO "if not exists" or something, to prevent error? */
ALTER TABLE responses
    ADD CONSTRAINT raw_f_len
    CHECK (cardinality(raw_f) > 1);

ALTER TABLE responses
    ADD CONSTRAINT df_over_f_len
    CHECK (cardinality(df_over_f) > 1);

ALTER TABLE responses
DROP CONSTRAINT responses_presentation_id_fkey,
ADD CONSTRAINT responses_presentation_id_fkey
    FOREIGN KEY (presentation_id)
    REFERENCES  presentations (presentation_id)
    ON DELETE CASCADE;

/* Would probably speed up inserts, but might be too risky. If db crashes, table
 * contents will probably be deleted. */
/* ALTER TABLE responses SET UNLOGGED; */


CREATE TABLE IF NOT EXISTS pid (
    -- mixture smallint REFERENCES mixtures (mixture) NOT NULL,
    odor1 smallint,
    odor2 smallint,

    recording_from timestamp REFERENCES recordings (started_at) NOT NULL,

    -- TODO positive / nonneg constraint. alt repr?
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

