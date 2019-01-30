
CREATE TABLE IF NOT EXISTS flies (
    /* TODO serial? work w/ pandas insert if just not specified? */
    /*fly SERIAL UNIQUE NOT NULL, */

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

    odor SERIAL UNIQUE NOT NULL,

    PRIMARY KEY(name, log10_conc_vv)
);

/* TODO or just make fk based pk w/ all fields of each odor duplicated? */
CREATE TABLE IF NOT EXISTS mixtures (
    /* TODO or just use name + conc? */
    odor1 integer REFERENCES odors (odor) NOT NULL,
    odor2 integer REFERENCES odors (odor) NOT NULL,
    PRIMARY KEY(odor1, odor2)
);

/* TODO table for stimulus info? */
/* TODO maybe stimulus code or version somewhere if nothing else? */

CREATE TABLE IF NOT EXISTS analysis_runs (
    analysis_description text NOT NULL,
    /* TODO precision? i think at least seconds? otherwise whatever is
     * convenient to work from time.time() in python... */
    analyzed_at timestamp NOT NULL,
    /* TODO actually any simpler making some kind of artificial key like this
     * for fk purposes? i didn't want to have to refer to two extra columns in
     * responses keys... */
    analysis_run SERIAL UNIQUE NOT NULL,

    /* TODO git version + unsaved changes? */
    /* TODO git remotes? */
    PRIMARY KEY(analysis_description, analyzed_at)
);

CREATE TABLE IF NOT EXISTS recordings (
    /*recording_num SERIAL PRIMARY KEY,*/
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
);

CREATE TABLE IF NOT EXISTS responses (
    /* TODO matter whether fk is an ID vs all columns of composite pk in other
     * table, as far as space / speed performance? */
    /*
    presentation_id integer REFERENCES presentations (presentation_id),
    */
    analysis integer REFERENCES analysis_runs (analysis_run) NOT NULL,

    prep_date timestamp,
    fly_num smallint,

    /*fly integer REFERENCES flies (fly) NOT NULL, */
    recording_from timestamp REFERENCES recordings (started_at) NOT NULL,

    /*
    mixture integer REFERENCES mixtures (mixture) NOT NULL,
    */
    odor1 integer,
    odor2 integer,

    /* TODO positive / nonneg constraint. alt repr? */
    repeat_num integer NOT NULL,

    /* rename? roi/footprint/component? */
    cell integer NOT NULL,

    from_onset double precision NOT NULL,
    df_over_f real NOT NULL,

    FOREIGN KEY(prep_date, fly_num) REFERENCES flies(prep_date, fly_num),
    FOREIGN KEY(odor1, odor2) REFERENCES mixtures(odor1, odor2),
    PRIMARY KEY(analysis, prep_date, fly_num, recording_from, cell,
        odor1, odor2, repeat_num, from_onset)
    /*
    PRIMARY KEY(analysis, fly, recording_from, cell, mixture, repeat_num,
        from_onset)
    */
    /*PRIMARY KEY(presentation_id, from_onset)*/
);

CREATE TABLE IF NOT EXISTS pid (
    /*
    mixture integer REFERENCES mixtures (mixture) NOT NULL,
    */
    odor1 integer,
    odor2 integer,

    recording_from timestamp REFERENCES recordings (started_at) NOT NULL,
    /* TODO positive / nonneg constraint. alt repr? */
    repeat_num integer NOT NULL,

    from_onset double precision NOT NULL,
    pid_out real NOT NULL,

    FOREIGN KEY(odor1, odor2) REFERENCES mixtures(odor1, odor2),
    PRIMARY KEY(odor1, odor2, recording_from, repeat_num, from_onset)
    /*
    PRIMARY KEY(mixture, recording_from, repeat_num, from_onset)
    */
);
/* TODO link to table w/ PID / flow settings or just include for each row here?
 * */

/* TODO worth keeping a representation of traces across whole experiment?
   convert everything to odor responses?

   just include start frame / time for each chunk?
 */

/* TODO store footprints? how to represent? */

/* TODO maybe store info about the analysis that generated the given responses
   as well as the source data */
