<?php
/**
 * flexipipe in TEITOK (action=flexipipe): the router. Installed into the TEITOK shared project by
 * install-teitok-modules.pl (Sources/ for the .php files, Scripts/ for .js / .css).
 *
 * - flexipipe_document.php: one document (cid): what it contains and what a run would do
 *   (act=status), running flexipipe on it (act=process; existing annotation kept unless replaced);
 * - flexipipe_setup.php: the corpus defaults, backends, models, profiles, install / upgrade;
 * - flexipipe_jobs.php: background jobs (act=jobs, act=job, act=jobcancel; ?api=job): training, long runs;
 * - flexipipe_train.php: training models on the corpus (act=training; trainstart, usemodel, mainmodel, delmodel);
 * - flexipipe_publish.php: publishing a corpus model (act=publish: the package, then flexipipe-hub's deposit form; publishget);
 * - flexipipe_legacy.php: the pages not rebuilt yet (act=tag: trying models on a text, the gold
 *   comparison, act=train) — the earlier module, until they are.
 *
 * Shared code: flexipipe_settings.php (settings.xml, as flexipipe reads it), flexipipe_cmd.php
 * (running flexipipe: escaped arguments, document paths, manifests, cached info calls).
 */

check_login();

$fp_act = (string)( $act ?? ( $_GET['act'] ?? '' ) );
$fp_cid = (string)( $_GET['cid'] ?? ( $_GET['id'] ?? '' ) );
// one document: its status, and running flexipipe on it (flexipipe_document.php)
$fp_document = $fp_cid !== '' && in_array( $fp_act, array( '', 'status', 'process', 'preview' ), true );
$fp_setup_acts = array( '', 'save', 'models', 'profiles', 'install', 'upgrade', 'modinst', 'refresh' );
$fp_train_acts = array( 'training', 'trainstart', 'usemodel', 'mainmodel', 'delmodel' );
$fp_job_acts = array( 'jobs', 'job', 'jobcancel' );
$fp_publish_acts = array( 'publish', 'publishget' );
$fp_api = isset( $_GET['api'] ) ? (string)$_GET['api'] : '';

// the pages not rebuilt yet (and their own data calls, ?api=flexipipe_…)
$fp_jobs_page = in_array( $fp_act, $fp_job_acts, true ) || $fp_api === 'job';
$fp_training_page = in_array( $fp_act, $fp_train_acts, true );
$fp_publish_page = in_array( $fp_act, $fp_publish_acts, true );
if ( !$fp_document && !$fp_jobs_page && !$fp_training_page && !$fp_publish_page && ( !in_array( $fp_act, $fp_setup_acts, true ) || strpos( $fp_api, 'flexipipe_' ) === 0 ) && file_exists( __DIR__ . '/flexipipe_legacy.php' ) ) {
	require __DIR__ . '/flexipipe_legacy.php';
	return;
}

require_once "$ttroot/common/Sources/venv.php";
require_once __DIR__ . '/flexipipe_settings.php';
require_once __DIR__ . '/flexipipe_cmd.php';
require_once __DIR__ . '/flexipipe_setup.php';
require_once __DIR__ . '/flexipipe_document.php';
require_once __DIR__ . '/flexipipe_jobs.php';
require_once __DIR__ . '/flexipipe_train.php';
require_once __DIR__ . '/flexipipe_publish.php';

/** A script or stylesheet of this module, inline (installed: ../Scripts/, in the repo: next to this file). */
function flexipipe_script( $name ) {
	foreach ( array( __DIR__ . "/../Scripts/$name", __DIR__ . "/$name" ) as $f ) {
		if ( !file_exists( $f ) ) continue;
		$code = file_get_contents( $f );
		return substr( $name, -4 ) === '.css' ? "<style>\n$code\n</style>" : "<script>\n$code\n</script>";
	}
	return '';
}

$venv = new VENV();
$fp_settings = new FlexipipeSettings();
$fp_cmd = new FlexipipeCmd( $venv );
$fp_jobs = new FlexipipeJobs( $fp_cmd );
$fp_is_admin = ( $user['permissions'] ?? '' ) === 'admin';
if ( $fp_api === 'job' ) flexipipe_jobs_api( $fp_jobs, $fp_is_admin );
if ( $fp_act === 'publishget' ) { $maintext .= flexipipe_publish_download( $fp_cmd, $fp_is_admin ); return; }  // the package (exits)

// [VENV] debug lines would end up in the page
$fp_debug = $debug ?? null;
$debug = 0;
$maintext .= flexipipe_script( 'flexipipe.css' ) . ( $fp_jobs_page
	? flexipipe_jobs_page( $fp_jobs, $action, $fp_act, $fp_is_admin )
	: ( $fp_publish_page ? flexipipe_publish_page( $fp_cmd, $action, $fp_is_admin ) : ( $fp_training_page ? flexipipe_training_page( $fp_settings, $fp_cmd, $fp_jobs, $action, $fp_act, $fp_is_admin ) : ( $fp_document
		? flexipipe_document_page( $fp_settings, $fp_cmd, $action, $fp_act, $fp_cid )
		: flexipipe_setup_page( $fp_settings, $fp_cmd, $venv, $action, $fp_act ) ) ) ) );
if ( $fp_debug !== null ) $debug = $fp_debug;
