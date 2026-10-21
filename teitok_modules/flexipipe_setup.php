<?php
/**
 * flexipipe set-up pages (action=flexipipe): the corpus defaults (language, model, per-language
 * models, options), the backends, the models, the profiles; install / upgrade of flexipipe in
 * TEITOK's Python environment. Included by flexipipe.php, which sets $fp_settings and $fp_cmd.
 * Data for the forms: ?api=languages|models|backends (JSON, through FlexipipeCmd's cache).
 */

function flexipipe_h( $s ) { return htmlspecialchars( (string)$s, ENT_QUOTES ); }

/** The sentence layouts (flexipipe's sentence_layout presets), with what they are for. */
function flexipipe_layouts() {
	return array(
		'written' => 'Written text: sentences stay within paragraphs; inline elements are split around them',
		'conversation' => 'Conversation: a speaker turn (u) ends a sentence; sentences are split inside timed segments (seg)',
		'subtitles' => 'Subtitles: sentences run across screens (u, seg); they are split inside them',
		'subtitles-standoff' => 'Subtitles, as token lists: sentences across screens are listed, not marked up',
	);
}

/** The layers flexipipe writes, as users know them. */
function flexipipe_layer_names() {
	return array( 'xpos' => 'Tags', 'lemma' => 'Lemmas', 'reg' => 'Normalized forms', 'expan' => 'Expansions',
		'trslit' => 'Transliterations', 'pron' => 'Pronunciation (from the text)' );
}

/** "Where flexipipe writes": per layer automatic (the corpus's name, shown) or an attribute chosen. */
function flexipipe_layers_rows( $fp_settings, $fp_cmd ) {
	$h = 'flexipipe_h';
	$opts = $fp_cmd->json( array( 'info', 'training', '--corpus', '.', '--output-format', 'json' ), 600 );
	$writes = $opts['writes'] ?? array();
	$set = $fp_settings->layer_attributes();
	$attrs = array_keys( $fp_settings->token_attributes() );
	$rows = '';
	foreach ( flexipipe_layer_names() as $layer => $label ) {
		$w = $writes[ $layer ] ?? array();
		$auto = $w['automatic'] ?? $layer;
		$cur = $set[ $layer ] ?? '';
		$names = array_values( array_unique( array_merge( $w['names'] ?? array( $layer ), $w['in_corpus'] ?? array(), $attrs ) ) );
		$names = array_values( array_diff( $names, array( 'form', 'id', 'ord', 'head', 'deprel', 'join' ) ) );
		$sel = "<select name='layer[$layer]'><option value=''>automatic (now @" . $h( $auto ) . ")</option>";
		foreach ( $names as $n ) $sel .= "<option value='" . $h( $n ) . "'" . ( $cur === $n ? ' selected' : '' ) . ">@" . $h( $n ) . "</option>";
		$sel .= "<option value='-'" . ( $cur === '-' ? ' selected' : '' ) . ">leave out (never written)</option></select>";
		$custom = $cur !== '' && $cur !== '-' && !in_array( $cur, $names, true ) ? $cur : '';
		$rows .= "<tr><th>" . $h( $label ) . "<td>$sel or <input name='layer_other[$layer]' size='10' placeholder='other name' value='" . $h( $custom ) . "'></tr>";
	}
	return $rows;
}

/** The steps another backend can do, as users know them, with what the model is. */
function flexipipe_step_names() {
	return array(
		'ner' => array( 'Named entities', 'a NameTag model, e.g. czech-cnec2.0 (empty: NameTag\'s model for the language)' ),
		'normalize' => array( 'Normalization', 'a Hunspell dictionary, e.g. cs_CZ' ),
		'pron' => array( 'Pronunciation (G2P)', 'the folder of a pronunciation model (model_pron.json), e.g. trained on this corpus with flexipipe train --pron' ),
	);
}

/** The Steps rows of the defaults form: per step a backend (or none) and a model. */
function flexipipe_steps_rows( $fp_settings ) {
	$routes = $fp_settings->task_routes();
	$rows = '';
	foreach ( flexipipe_step_names() as $task => list( $label, $model_help ) ) {
		$cur = $routes[ $task ] ?? array( 'backend' => '', 'model' => '' );
		$sel = "<select name='step_backend[$task]'><option value=''>The main backend's, if it does this step</option>";
		foreach ( FlexipipeSettings::TASKS[ $task ] as $b ) {
			$sel .= "<option value='" . flexipipe_h( $b ) . "'" . ( strcasecmp( $cur['backend'], $b ) === 0 ? ' selected' : '' ) . ">" . flexipipe_h( $b ) . "</option>";
		}
		if ( $cur['backend'] !== '' && !in_array( strtolower( $cur['backend'] ), FlexipipeSettings::TASKS[ $task ], true ) ) {
			$sel .= "<option value='" . flexipipe_h( $cur['backend'] ) . "' selected>" . flexipipe_h( $cur['backend'] ) . "</option>";
		}
		$sel .= "</select>";
		$rows .= "<tr><th>" . flexipipe_h( $label ) . "<td>$sel <input name='step_model[$task]' size='30' value='" . flexipipe_h( $cur['model'] ) . "' placeholder='model'>"
			. "<td class='fp-help'>Model: " . flexipipe_h( $model_help ) . ".";
	}
	return $rows;
}

function flexipipe_setup_api( $fp_cmd, $api ) {
	$out = array( 'ok' => false );
	if ( $api === 'languages' ) {
		$data = $fp_cmd->json( array( 'info', 'languages', '--output-format', 'json' ) );
		$list = array();
		foreach ( $data['languages'] ?? array() as $e ) {
			$code = $e['iso_639_1'] ?? ( $e['iso_639_3'] ?? '' );
			if ( $code !== '' ) $list[] = array( 'code' => $code, 'name' => $e['name'] ?? $code );
		}
		usort( $list, function ( $a, $b ) { return strcasecmp( $a['name'], $b['name'] ); } );
		$out = array( 'ok' => $data !== null, 'languages' => $list );
	} elseif ( $api === 'models' ) {
		$lang = trim( (string)( $_GET['language'] ?? '' ) );
		$list = array();
		$auto = null;
		if ( $lang !== '' ) {
			$data = $fp_cmd->json( array( 'info', 'models', '--language', $lang, '--output-format', 'json' ) );
			foreach ( $data['models'] ?? array() as $e ) {
				if ( empty( $e['backend'] ) || empty( $e['model'] ) ) continue;
				$list[] = array( 'backend' => $e['backend'], 'model' => $e['model'], 'installed' => $e['installed'] ?? null );
			}
			$auto = $data['auto'] ?? null;  // what flexipipe would pick now
			// the models trained on this corpus first (flexitag, by their folder)
			$own = array();
			foreach ( flexipipe_corpus_models( $fp_cmd ) as $m ) {
				$ml = strtolower( (string)( $m['language'] ?? ( $m['pron']['language'] ?? '' ) ) );
				if ( $ml !== '' && $ml !== strtolower( $lang ) ) continue;
				$own[] = array( 'backend' => 'flexitag', 'model' => $m['folder'], 'installed' => true,
					'label' => 'trained on this corpus, ' . ( $m['created'] ?? '' ) . ( !empty( $m['in_use'] ) ? ' (in use)' : '' ) );
			}
			$list = array_merge( $own, $list );
		}
		$out = array( 'ok' => true, 'models' => $list, 'auto' => $auto );
	} elseif ( $api === 'backends' ) {
		$data = $fp_cmd->json( array( 'info', 'backends', '--output-format', 'json' ) );
		$out = array( 'ok' => $data !== null, 'backends' => $data['backends'] ?? array() );
	}
	header( 'Content-Type: application/json' );
	echo json_encode( $out );
	exit;
}

function flexipipe_setup_save( $fp_settings ) {
	$per = array();
	foreach ( (array)( $_POST['lang_code'] ?? array() ) as $i => $code ) {
		$code = trim( (string)$code );
		if ( $code === '' ) continue;
		$per[ $code ] = array( 'backend' => trim( (string)( $_POST['lang_backend'][ $i ] ?? '' ) ), 'model' => trim( (string)( $_POST['lang_model'][ $i ] ?? '' ) ) );
	}
	$flags = array();
	foreach ( FlexipipeSettings::FLAGS as $f ) $flags[ $f ] = ( $_POST[ $f ] ?? '' ) === 'yes';
	$layers = array();
	foreach ( array_keys( flexipipe_layer_names() ) as $layer ) {
		$other = trim( (string)( $_POST['layer_other'][ $layer ] ?? '' ) );
		$chosen = $other !== '' ? $other : trim( (string)( $_POST['layer'][ $layer ] ?? '' ) );
		if ( $chosen !== '' && $chosen !== '-' && !preg_match( '/^[A-Za-z_][A-Za-z0-9_.-]*$/', $chosen ) ) $chosen = '';  // an XML attribute name
		$layers[ $layer ] = $chosen;
	}
	$fp_settings->set_layer_attributes( $layers );
	$steps = array();
	foreach ( array_keys( FlexipipeSettings::TASKS ) as $task ) {
		$steps[ $task ] = array( 'backend' => trim( (string)( $_POST['step_backend'][ $task ] ?? '' ) ), 'model' => trim( (string)( $_POST['step_model'][ $task ] ?? '' ) ) );
	}
	$fp_settings->set_task_routes( $steps );
	$layout = trim( (string)( $_POST['layout'] ?? '' ) );
	if ( $layout === '' || isset( flexipipe_layouts()[ $layout ] ) ) $fp_settings->set_layout_preset( $layout );
	$fp_settings->set_defaults(
		trim( (string)( $_POST['default_lang'] ?? '' ) ),
		trim( (string)( $_POST['default_backend'] ?? '' ) ),
		trim( (string)( $_POST['default_model'] ?? '' ) ),
		$flags,
		$per
	);
	return $fp_settings->save();
}

function flexipipe_backends_table( $fp_cmd, $action ) {
	$data = $fp_cmd->json( array( 'info', 'backends', '--output-format', 'json' ) );
	if ( !$data ) return "<p class=wrong>The list of backends could not be read from flexipipe.</p>";
	$labels = flexipipe_step_labels();
	$out = "<table class='fp-table'><thead><tr><th>Backend<th>Steps<th>Description<th>Status</tr></thead><tbody>";
	foreach ( $data['backends'] ?? array() as $b ) {
		$available = ( $b['available'] ?? false ) === true;
		$name = flexipipe_h( $b['name'] ?? ( $b['backend'] ?? '' ) );
		if ( !empty( $b['url'] ) ) $name = "<a href='" . flexipipe_h( $b['url'] ) . "' target='_blank' rel='noopener'>$name</a>";
		$status = flexipipe_h( $b['status'] ?? ( $available ? 'available' : '' ) );
		if ( !$available && !empty( $b['install_hint'] ) && preg_match( '/^\s*pip\s+install\s+(.+)$/i', $b['install_hint'], $m ) ) {
			$module = trim( $m[1], " \t\"'" );
			$status .= " (<a href='index.php?action=$action&act=modinst&module=" . rawurlencode( $module ) . "'>install</a>)";
		}
		// what it does (xpos counts as tagging); * only with some of its models
		$steps = array();
		foreach ( (array)( $b['tasks'] ?? array() ) as $t ) {
			if ( $t === 'xpos' ) continue;
			$steps[] = flexipipe_h( $labels[ $t ] ?? $t ) . ( in_array( $t, (array)( $b['tasks_depend_on_model'] ?? array() ), true ) ? '*' : '' );
		}
		$steps = $steps ? implode( ', ', $steps ) : '';
		if ( !empty( $b['trainable'] ) ) $steps .= " <span class='fp-help'>(can be trained on a corpus)</span>";
		$out .= "<tr" . ( $available ? '' : " class='fp-off'" ) . "><td>$name<td>$steps<td>" . flexipipe_h( $b['description'] ?? '' ) . "<td>$status</tr>";
	}
	return $out . "</tbody></table><p class='fp-help'>* only with some of its models (a flexitag model trained with pronunciations, a UDPipe model with a parser …).</p>";
}

function flexipipe_setup_page( $fp_settings, $fp_cmd, $venv, $action, $act ) {
	global $user;
	$is_admin = ( $user['permissions'] ?? '' ) === 'admin';
	$out = "<h1>NLP pipeline: flexipipe</h1>";

	if ( isset( $_GET['api'] ) ) flexipipe_setup_api( $fp_cmd, (string)$_GET['api'] );

	if ( $act === 'install' || $act === 'upgrade' ) {
		$flag = $act === 'upgrade' ? '--upgrade ' : '';
		shell_exec( 'FLEXIPIPE_QUIET_INSTALL=1 ' . escapeshellarg( $venv->venvdir . '/bin/python' ) . " -m pip install {$flag}git+https://github.com/ufal/flexipipe.git >> tmp/venv-install.log 2>&1 &" );
		FlexipipeCmd::clear_cache();
		return $out . "<p>flexipipe is being " . ( $act === 'upgrade' ? 'upgraded' : 'installed' ) . " in the background; this can take a few minutes. <a href='index.php?action=$action'>Check again</a></p>";
	}
	if ( !$venv->checkmod( 'flexipipe' ) ) {
		return $out . "<p>flexipipe is not installed in TEITOK's Python environment yet. "
			. ( $is_admin ? "<a href='index.php?action=$action&act=install'>Install flexipipe</a>" : "An administrator can install it here." ) . "</p>";
	}
	if ( $fp_cmd->version() === '' ) {
		// installed (pip knows it) but it does not start: a broken or moved installation
		list( , , $err ) = $fp_cmd->run( array( '--version' ) );
		$last = trim( (string)preg_replace( '/^.*\n/s', '', trim( $err ) ) );
		return $out . "<p class=wrong>flexipipe is installed in TEITOK's Python environment but does not start there, so the settings cannot be shown.</p>"
			. ( $is_admin
				? "<p><a href='index.php?action=$action&act=upgrade'>Reinstall flexipipe</a></p>"
					. "<p class='fp-help'>" . flexipipe_h( $venv->venvdir ) . ( $last !== '' ? ': ' . flexipipe_h( $last ) : '' ) . "</p>"
				: "<p>An administrator can reinstall it here.</p>" );
	}
	if ( $act === 'refresh' && $is_admin ) {
		FlexipipeCmd::clear_cache();
		$out .= "<p class=right>The lists of languages, models and backends will be read from flexipipe again.</p>";
	}
	if ( $act === 'modinst' && $is_admin ) {
		$venv->installmod( (string)( $_GET['module'] ?? '' ), true );
		FlexipipeCmd::clear_cache();
		$out .= "<p>The backend is being installed in the background.</p>";
	}

	if ( $act === 'save' ) {
		if ( !$is_admin ) return $out . "<p class=wrong>Only administrators can change the settings.</p>";
		$problem = flexipipe_setup_save( $fp_settings );
		$out .= $problem === '' ? "<p class=right>The settings were saved.</p>" : "<p class=wrong>The settings were not saved: " . flexipipe_h( $problem ) . ".</p>";
		$fp_settings = new FlexipipeSettings();
	}

	if ( $act === 'models' ) {
		$data = $fp_cmd->json( array( 'info', 'models', '--output-format', 'json' ) );
		$out .= "<h2>Models</h2><p>Grey: not downloaded yet. Click a model to try it on a text.</p>
			<table class='fp-table' data-sortable><thead><tr><th>Backend<th>Model<th>Language<th>Description</tr></thead><tbody>";
		foreach ( $data['models'] ?? array() as $m ) {
			$off = ( $m['installed'] ?? true ) === false ? " class='fp-off'" : '';
			$q = http_build_query( array( 'action' => $action, 'act' => 'tag', 'backend' => $m['backend'] ?? '', 'model' => $m['model'] ?? '', 'language' => $m['language_iso'] ?? '' ) );
			$out .= "<tr$off><td>" . flexipipe_h( $m['backend'] ?? '' ) . "<td><a href='index.php?$q'>" . flexipipe_h( $m['model'] ?? '' ) . "</a><td>"
				. flexipipe_h( ( $m['language_name'] ?? '' ) . ' (' . ( $m['language_iso'] ?? '' ) . ')' ) . "<td>" . flexipipe_h( $m['description'] ?? '' ) . "</tr>";
		}
		return $out . "</tbody></table><p><a href='index.php?action=$action'>Back</a></p>";
	}

	if ( $act === 'profiles' ) {
		$titles = array( 'manifests' => 'Model manifests (which model applies to which documents)', 'applications' => 'Application profiles (how results are written back)', 'datasets' => 'Dataset profiles (training)' );
		foreach ( $titles as $group => $title ) {
			$items = $fp_settings->profiles( $group );
			$out .= "<h2>" . flexipipe_h( $title ) . "</h2>";
			if ( !$items ) { $out .= "<p>None defined.</p>"; continue; }
			$out .= "<table class='fp-table'><thead><tr><th>Key<th>Settings</tr></thead><tbody>";
			foreach ( $items as $key => $item ) {
				$parts = array();
				foreach ( $item as $k => $v ) if ( $k !== 'key' && $v !== '' ) $parts[] = flexipipe_h( $k ) . '=' . flexipipe_h( $v );
				$out .= "<tr><td><code>" . flexipipe_h( $key ) . "</code><td>" . implode( '; ', $parts ) . "</tr>";
			}
			$out .= "</tbody></table>";
		}
		return $out . "<p><a href='index.php?action=$action'>Back</a></p>";
	}

	// the overview: defaults form, backends
	$per = array();
	foreach ( $fp_settings->language_models() as $code => $bm ) $per[] = array( 'code' => $code, 'backend' => $bm['backend'], 'model' => $bm['model'] );
	$form = array(
		'default_lang' => $fp_settings->language(),
		'default_backend' => $fp_settings->get( 'backend' ),
		'default_model' => $fp_settings->get( 'model' ),
		'per_lang' => $per,
		'api' => "index.php?action=$action",
	);
	$sel = function ( $name ) use ( $fp_settings ) {
		$yes = $fp_settings->flag( $name );
		return "<select name='$name'><option value='no'" . ( $yes ? '' : ' selected' ) . ">No</option><option value='yes'" . ( $yes ? ' selected' : '' ) . ">Yes</option></select>";
	};
	$out .= "<p>flexipipe annotates the documents of this corpus: tokens, sentences, tags, lemmas, dependencies and more, with the model chosen for their language.
		To annotate a document, open it and choose flexipipe in its file management. <a href='index.php?action=$action&act=tag'>Try the models on a text</a>
		· <a href='index.php?action=$action&act=models'>All models</a> · <a href='index.php?action=$action&act=profiles'>Profiles</a>"
		. ( glob( FlexipipeJobs::DIR . '/*/job.json' ) ? " · <a href='index.php?action=$action&act=jobs'>Background jobs</a>" : '' ) . "</p>";
	$cur = $fp_settings->layout_preset();
	$layout_sel = "<select name='layout'><option value=''" . ( $cur === '' ? ' selected' : '' ) . ">Written text (flexipipe's default)</option>";
	foreach ( flexipipe_layouts() as $k => $label ) {
		if ( $k === 'written' ) continue;
		$layout_sel .= "<option value='" . flexipipe_h( $k ) . "'" . ( $cur === $k ? ' selected' : '' ) . ">" . flexipipe_h( $label ) . "</option>";
	}
	$layout_sel .= "</select>";
	$layout_extra = $cur !== '' && !isset( flexipipe_layouts()[ $cur ] ) ? "settings.xml names the layout " . flexipipe_h( $cur ) . ", which flexipipe does not know." : '';
	$disabled = $is_admin ? '' : ' disabled';
	$out .= "<h2>Defaults for this corpus</h2>
		<form method='post' action='index.php?action=$action&act=save' id='fp-defaults'><fieldset$disabled>
		<table class='fp-form'>
		<tr><th>Language<td><select name='default_lang' id='fp-lang'><option value=''>Detect for each document</option></select>
			<td class='fp-help'>Used when a document does not state its language.
		<tr><th>Model<td><select id='fp-model'><option value=''>flexipipe's choice</option></select>
			<input type='hidden' name='default_backend' id='fp-backend-in'><input type='hidden' name='default_model' id='fp-model-in'>
			<td class='fp-help'>For the language above.
		<tr><th>Download missing models<td>" . $sel( 'download-model' ) . "<td class='fp-help'>flexipipe may download a model it needs.
		<tr><th>Only this corpus's tags<td>" . $sel( 'known-tags-only' ) . "<td class='fp-help'>Write only the attributes defined in this corpus.
		<tr><th>Annotate the raw text<td>" . $sel( 'use-raw-text' ) . "<td class='fp-help'>Give the backend the running text rather than the existing tokens.
		<tr><th>Sentences<td>" . $layout_sel . "<td class='fp-help'>How sentences fit the structure of the documents (paragraphs, speaker turns, timed segments)."
		. ( $layout_extra ? "<br>" . $layout_extra : '' ) . "
		</table>
		<h3>Steps the model does not do</h3>
		<p class='fp-help'>Most models tokenize, tag, lemmatize and parse. For the steps below, choose the backend this corpus uses when its model does not do them; a step nobody does is not offered on the documents.</p>
		<table class='fp-form'>" . flexipipe_steps_rows( $fp_settings ) . "</table>
		<details id='fp-layers'><summary>Where flexipipe writes</summary>
		<p class='fp-help'>flexipipe writes each layer under the name this corpus already uses for it (tags in @tag, @pos or @msd; normalized forms in @nform …). Choose an attribute to decide yourself — for instance to keep @msd as it is and have the tags written to @xpos, or when an attribute with a familiar name holds something else. <i>Leave out</i>: the layer is never written (what is there stays), and a step that only produces it (pronunciation, normalization, lemmas) is not run.</p>
		<table class='fp-form'>" . flexipipe_layers_rows( $fp_settings, $fp_cmd ) . "</table></details>
		<h3>Models per language</h3>
		<table class='fp-form' id='fp-langs'><thead><tr><th>Language<th>Model<th></tr></thead><tbody></tbody></table>
		<p><button type='button' id='fp-add-lang'>Add a language</button> <button type='submit'>Save</button></p>
		</fieldset></form>";
	if ( !$is_admin ) $out .= "<p class='fp-help'>Only administrators can change these settings.</p>";
	$out .= "<script>var flexipipeSetup = " . json_encode( $form ) . ";</script>"
		. flexipipe_script( 'flexipipe-setup.js' );
	$out .= flexipipe_training_summary( $fp_cmd, new FlexipipeJobs( $fp_cmd ), $action );
	$out .= "<h2>Backends</h2>" . flexipipe_backends_table( $fp_cmd, $action );
	if ( $is_admin ) $out .= "<p class='fp-help'>flexipipe " . flexipipe_h( $fp_cmd->version() ) . " · <a href='index.php?action=$action&act=upgrade'>Upgrade</a>"
		. " · <a href='index.php?action=$action&act=refresh' title='The lists are kept for an hour; read them from flexipipe again now'>Refresh the lists</a></p>";
	return $out;
}
