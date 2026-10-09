<?php

	check_login();

	// Safety cap for gold CoNLL-U comparison so large uploads do not overwhelm PHP.
	if ( !defined('FLEXIPIPE_GOLD_MAX_SENTENCES') ) define('FLEXIPIPE_GOLD_MAX_SENTENCES', 200);
	if ( !defined('FLEXIPIPE_GOLD_MAX_TOKENS') ) define('FLEXIPIPE_GOLD_MAX_TOKENS', 5000);

	/**
	 * Parse CoNLL-U content into sentences (each sentence = array of tokens) and plain text.
	 * Captures # text = ... from comment lines (CoNLL-U requirement) into sentence_texts.
	 */
	function flexipipe_parse_conllu( $content, $maxSentences = null, $maxTokens = null ) {
		$sentences = [];
		$sentenceTexts = [];
		$current = [];
		$nextText = '';
		$tokenCount = 0;
		$truncated = false;
		$lines = preg_split('/\r\n|\r|\n/', $content);
		foreach ( $lines as $line ) {
			$lineTrim = trim($line);
			if ( $lineTrim === '' ) {
				if ( !empty($current) ) {
					$sentences[] = $current;
					$sentenceTexts[] = $nextText;
					$tokenCount += count($current);
					$current = [];
					$nextText = '';
					if (
						( $maxSentences !== null && count($sentences) >= (int)$maxSentences ) ||
						( $maxTokens !== null && $tokenCount >= (int)$maxTokens )
					) {
						$truncated = true;
						break;
					}
				}
				continue;
			}
			if ( $lineTrim[0] === '#' ) {
				if ( preg_match('/^#\s*text\s*=\s*(.*)$/u', $lineTrim, $m) ) {
					$nextText = trim($m[1]);
				}
				continue;
			}
			$cols = explode("\t", $lineTrim);
			if ( count($cols) < 8 ) continue;
			$id = $cols[0];
			if ( $id === '_' || $id === '' ) continue;
			if ( preg_match('/^\d+$/', $id ) || preg_match('/^\d+-\d+$/', $id ) ) {
				$current[] = [
					'form'   => isset($cols[1]) ? $cols[1] : '_',
					'lemma'  => isset($cols[2]) ? $cols[2] : '_',
					'upos'   => isset($cols[3]) ? $cols[3] : '_',
					'xpos'   => isset($cols[4]) ? $cols[4] : '_',
					'feats'  => isset($cols[5]) ? $cols[5] : '_',
					'head'   => isset($cols[6]) ? $cols[6] : '_',
					'deprel' => isset($cols[7]) ? $cols[7] : '_',
				];
			}
		}
		if ( !empty($current) ) {
			$sentences[] = $current;
			$sentenceTexts[] = $nextText;
			$tokenCount += count($current);
			if (
				( $maxSentences !== null && count($sentences) >= (int)$maxSentences ) ||
				( $maxTokens !== null && $tokenCount >= (int)$maxTokens )
			) {
				$truncated = true;
			}
		}
		$textLines = [];
		foreach ( $sentences as $sent ) {
			$forms = array_map(function ($t) { return $t['form']; }, $sent);
			$textLines[] = implode(' ', $forms);
		}
		return [
			'sentences' => $sentences,
			'sentence_texts' => $sentenceTexts,
			'text' => implode("\n", $textLines),
			'token_count' => $tokenCount,
			'truncated' => $truncated,
		];
	}

	/**
	 * Compare gold vs model CoNLL-U sentences (same order). Returns counts for accuracy.
	 * Includes xpos (when gold has xpos) and upos+feats (when gold has both).
	 */
	function flexipipe_compare_conllu( $goldSentences, $modelSentences ) {
		$upos_ok = $upos_total = $lemma_ok = $lemma_total = 0;
		$xpos_ok = $xpos_total = $upos_feats_ok = $upos_feats_total = 0;
		$uas_ok = $uas_total = $las_ok = $las_total = 0;
		$skipped = 0;
		$n = min(count($goldSentences), count($modelSentences));
		for ( $s = 0; $s < $n; $s++ ) {
			$g = $goldSentences[$s];
			$m = isset($modelSentences[$s]) ? $modelSentences[$s] : [];
			if ( count($g) !== count($m) ) {
				$skipped++;
				continue;
			}
			for ( $i = 0; $i < count($g); $i++ ) {
				$gt = $g[$i];
				$mt = $m[$i];
				if ( isset($gt['upos']) && $gt['upos'] !== '_' && $gt['upos'] !== '' ) {
					$upos_total++;
					if ( isset($mt['upos']) && strcasecmp(trim($gt['upos']), trim($mt['upos'])) === 0 ) $upos_ok++;
				}
				if ( isset($gt['lemma']) && $gt['lemma'] !== '_' && $gt['lemma'] !== '' ) {
					$lemma_total++;
					if ( isset($mt['lemma']) && strcasecmp(trim($gt['lemma']), trim($mt['lemma'])) === 0 ) $lemma_ok++;
				}
				if ( isset($gt['xpos']) && $gt['xpos'] !== '_' && $gt['xpos'] !== '' ) {
					$xpos_total++;
					if ( isset($mt['xpos']) && strcasecmp(trim($gt['xpos']), trim($mt['xpos'])) === 0 ) $xpos_ok++;
				}
				$uposG = isset($gt['upos']) ? trim($gt['upos']) : '_';
				$featsG = isset($gt['feats']) ? trim($gt['feats']) : '_';
				if ( $uposG !== '_' && $uposG !== '' && $featsG !== '_' && $featsG !== '' ) {
					$upos_feats_total++;
					$uposM = isset($mt['upos']) ? trim($mt['upos']) : '_';
					$featsM = isset($mt['feats']) ? trim($mt['feats']) : '_';
					if ( strcasecmp($uposG, $uposM) === 0 && strcasecmp($featsG, $featsM) === 0 ) $upos_feats_ok++;
				}
				$hG = isset($gt['head']) ? trim($gt['head']) : '_';
				$hM = isset($mt['head']) ? trim($mt['head']) : '_';
				$dG = isset($gt['deprel']) ? trim($gt['deprel']) : '_';
				$dM = isset($mt['deprel']) ? trim($mt['deprel']) : '_';
				if ( $hG !== '_' && $hG !== '' && is_numeric($hG) ) {
					$uas_total++;
					$las_total++;
					if ( $hM !== '_' && $hM !== '' && (int)$hG === (int)$hM ) {
						$uas_ok++;
						if ( $dM !== '_' && $dM !== '' && strcasecmp($dG, $dM) === 0 ) $las_ok++;
					}
				}
			}
		}
		return [
			'upos_ok' => $upos_ok, 'upos_total' => $upos_total,
			'lemma_ok' => $lemma_ok, 'lemma_total' => $lemma_total,
			'xpos_ok' => $xpos_ok, 'xpos_total' => $xpos_total,
			'upos_feats_ok' => $upos_feats_ok, 'upos_feats_total' => $upos_feats_total,
			'uas_ok' => $uas_ok, 'uas_total' => $uas_total,
			'las_ok' => $las_ok, 'las_total' => $las_total,
			'sentences_skipped' => $skipped,
		];
	}

	/**
	 * Build HTML for token-by-token comparison (gold vs model). Highlights diffs.
	 * Uses # text when provided (CoNLL-U); for mismatches shows both gold and model sentence text.
	 */
	function flexipipe_gold_compare_html( $goldSentences, $modelSentences, $goldTexts = [], $modelTexts = [] ) {
		$out = '';
		$n = min(count($goldSentences), count($modelSentences));
		for ( $s = 0; $s < $n; $s++ ) {
			$g = $goldSentences[$s];
			$m = isset($modelSentences[$s]) ? $modelSentences[$s] : [];
			$goldSentText = ( isset($goldTexts[$s]) && $goldTexts[$s] !== '' ) ? $goldTexts[$s] : implode(' ', array_map(function ($t) { return isset($t['form']) ? $t['form'] : ''; }, $g));
			$modelSentText = ( isset($modelTexts[$s]) && $modelTexts[$s] !== '' ) ? $modelTexts[$s] : implode(' ', array_map(function ($t) { return isset($t['form']) ? $t['form'] : ''; }, $m));
			if ( count($g) !== count($m) ) {
				$out .= '<p class="fp-gold-sent-mismatch"><strong>Sentence ' . ($s + 1) . ':</strong> token count mismatch (gold ' . count($g) . ', model ' . count($m) . ').</p>';
				$out .= '<p class="fp-gold-sent-text"><strong>Gold:</strong> ' . htmlspecialchars($goldSentText) . '</p>';
				$out .= '<p class="fp-gold-sent-text"><strong>Model:</strong> ' . htmlspecialchars($modelSentText) . '</p>';
				continue;
			}
			$out .= '<p class="fp-gold-sent-text"><strong>Sentence ' . ($s + 1) . ':</strong> ' . htmlspecialchars($goldSentText) . '</p>';
			$out .= '<table class="fp-gold-compare-table"><thead><tr><th>#</th><th>Form</th><th>Lemma</th><th>UPOS</th><th>XPOS</th><th>Feats</th><th>Head</th><th>Deprel</th></tr></thead><tbody>';
			foreach ( $g as $i => $gt ) {
				$mt = isset($m[$i]) ? $m[$i] : [];
				$cells = ['lemma', 'upos', 'xpos', 'feats', 'head', 'deprel'];
				$row = '<tr><td>' . ($i + 1) . '</td><td>' . htmlspecialchars($gt['form'] ?? '') . '</td>';
				foreach ( $cells as $key ) {
					$gv = isset($gt[$key]) ? trim((string)$gt[$key]) : '_';
					$mv = isset($mt[$key]) ? trim((string)$mt[$key]) : '_';
					$same = (string)$gv === (string)$mv || strcasecmp($gv, $mv) === 0;
					$cellClass = $same ? '' : ' class="fp-gold-diff"';
					$display = $same ? htmlspecialchars($gv) : htmlspecialchars($gv) . ' → ' . htmlspecialchars($mv);
					$row .= '<td' . $cellClass . '>' . $display . '</td>';
				}
				$row .= '</tr>';
				$out .= $row;
			}
			$out .= '</tbody></table>';
		}
		return $out;
	}

	/**
	 * Read a flexipipe profile registry item via getset("flexipipe/$group/$key").
	 */
	function flexipipe_get_profile_item( $group, $key ) {
		if ( $key === '' || $key === null ) return null;
		$item = getset( "flexipipe/$group/$key" );
		return ( is_array( $item ) && !empty( $item ) ) ? $item : null;
	}

	/**
	 * List all items in a flexipipe profile registry group.
	 */
	function flexipipe_list_profile_items( $group ) {
		$items = getset( "flexipipe/$group", array() );
		return is_array( $items ) ? $items : array();
	}

	function flexipipe_profile_path( $group, $key ) {
		$item = flexipipe_get_profile_item( $group, $key );
		if ( !$item || empty( $item['path'] ) ) return null;
		return $item['path'];
	}

	function flexipipe_match_recond( $pattern, $path ) {
		if ( $pattern === '' || $pattern === null ) return true;
		return @preg_match( '#' . $pattern . '#u', $path ) === 1;
	}

	function flexipipe_match_xpres( $xpathExpr, $xmlContent ) {
		if ( $xpathExpr === '' || $xpathExpr === null ) return true;
		if ( !is_string( $xmlContent ) || $xmlContent === '' ) return false;
		libxml_use_internal_errors( true );
		$doc = new DOMDocument();
		if ( !$doc->loadXML( $xmlContent ) ) return false;
		$xp = new DOMXPath( $doc );
		$root = $doc->documentElement;
		if ( $root && $root->namespaceURI ) {
			$xp->registerNamespace( 'tei', $root->namespaceURI );
		}
		$nodes = @$xp->query( $xpathExpr );
		return ( $nodes && $nodes->length > 0 );
	}

	/**
	 * Resolve manifest by explicit key or by xpres/recond match on file path + XML.
	 */
	function flexipipe_resolve_manifest( $cidPath, $xmlContent, $manifestKey = null ) {
		if ( $manifestKey ) {
			$item = flexipipe_get_profile_item( 'manifests', $manifestKey );
			return $item ? array( $manifestKey, $item ) : array( null, null );
		}
		foreach ( flexipipe_list_profile_items( 'manifests' ) as $key => $item ) {
			if ( !is_array( $item ) ) continue;
			$recond = isset( $item['recond'] ) ? $item['recond'] : '';
			$xpres = isset( $item['xpres'] ) ? $item['xpres'] : '';
			if ( flexipipe_match_recond( $recond, $cidPath ) && flexipipe_match_xpres( $xpres, $xmlContent ) ) {
				return array( $key, $item );
			}
		}
		return array( null, null );
	}

	/**
	 * Build flexipipe process CLI args for a TEITOK XML file.
	 */
	function flexipipe_build_process_cmd( $filename, $manifestItem = null, $applicationPath = null, $extra = array() ) {
		$cmd = "-m flexipipe process --teitok --input " . escapeshellarg( $filename );
		if ( is_array( $manifestItem ) ) {
			if ( !empty( $manifestItem['backend'] ) ) {
				$cmd .= " --backend " . escapeshellarg( $manifestItem['backend'] );
			}
			if ( !empty( $manifestItem['model'] ) ) {
				$cmd .= " --model " . escapeshellarg( $manifestItem['model'] );
			}
			// Optional portable bundle manifest file (not used for normal TEITOK registry items).
			if ( !empty( $manifestItem['bundle'] ) && file_exists( $manifestItem['bundle'] ) ) {
				$cmd .= " --manifest " . escapeshellarg( $manifestItem['bundle'] );
			} elseif ( !empty( $manifestItem['path'] ) && file_exists( $manifestItem['path'] ) ) {
				$cmd .= " --manifest " . escapeshellarg( $manifestItem['path'] );
			}
			if ( !empty( $manifestItem['tasks'] ) ) {
				$cmd .= " --tasks " . escapeshellarg( $manifestItem['tasks'] );
			}
		}
		if ( $applicationPath && file_exists( $applicationPath ) ) {
			$cmd .= " --application " . escapeshellarg( $applicationPath );
		}
		if ( !empty( $extra['language'] ) ) {
			$cmd .= " --language " . escapeshellarg( $extra['language'] );
		}
		if ( !empty( $extra['tokenize'] ) ) {
			$cmd .= " --tokenize";
		} else {
			$cmd .= " --tokenize";
		}
		if ( !empty( $extra['writeback'] ) || !isset( $extra['writeback'] ) ) {
			$cmd .= " --writeback";
		}
		if ( !empty( $extra['verbose'] ) ) {
			$cmd .= " --verbose";
		}
		return $cmd;
	}

	function flexipipe_profiles_table_html( $group, $title ) {
		$items = flexipipe_list_profile_items( $group );
		if ( empty( $items ) ) return '';
		$out = "<h3>" . htmlspecialchars( $title ) . "</h3><table class='fp-profiles'><thead><tr><th>Key<th>Path<th>Details</thead><tbody>";
		foreach ( $items as $key => $item ) {
			if ( !is_array( $item ) ) continue;
			$path = isset( $item['path'] ) ? $item['path'] : '';
			if ( $path === '' && !empty( $item['bundle'] ) ) $path = $item['bundle'];
			$details = array();
			if ( !empty( $item['label'] ) ) $details[] = htmlspecialchars( $item['label'] );
			if ( $group === 'manifests' ) {
				if ( !empty( $item['backend'] ) ) $details[] = 'backend=' . htmlspecialchars( $item['backend'] );
				if ( !empty( $item['model'] ) ) $details[] = 'model=' . htmlspecialchars( $item['model'] );
				if ( !empty( $item['xpres'] ) ) $details[] = 'xpres=' . htmlspecialchars( $item['xpres'] );
				if ( !empty( $item['recond'] ) ) $details[] = 'recond=' . htmlspecialchars( $item['recond'] );
			}
			$out .= "<tr><td><code>" . htmlspecialchars( $key ) . "</code><td>" . htmlspecialchars( $path ) . "<td>" . implode( '; ', $details ) . "</tr>";
		}
		$out .= "</tbody></table>";
		return $out;
	}

	$cid = $_GET['cid'] or $cid = $_GET['id'];

	# Check if flexipipe is installed
	require("$ttroot/common/Sources/venv.php");
	$venv = new VENV();

	# API: same as EasyCorp (languages / models / backends) so the form can use the same AJAX. Use shell_exec to avoid [VENV] in output.
	if ( isset($_GET['api']) && in_array($_GET['api'], ['flexipipe_languages', 'flexipipe_models', 'flexipipe_backends', 'flexipipe_profiles'], true) ) {
		$py = $venv->venvdir . '/bin/python';
		$api = $_GET['api'];
		if ( $api === 'flexipipe_backends' ) {
			$cmd = escapeshellarg($py) . ' -m flexipipe info backends --output-format json 2>&1';
			$json = shell_exec($cmd);
			$data = $json ? json_decode(trim($json), true) : null;
			$out = ['ok' => false, 'api' => 'flexipipe_backends', 'backends' => []];
			if ( is_array($data) && !empty($data['backends']) ) {
				$list = [];
				foreach ( $data['backends'] as $e ) {
					if ( !is_array($e) ) continue;
					$be = $e['backend'] ?? $e['name'] ?? null;
					$name = $e['name'] ?? $e['backend'] ?? $be;
					if ( $be !== null && $be !== '' ) $list[] = ['backend' => (string)$be, 'name' => (string)$name];
				}
				$out = ['ok' => true, 'api' => 'flexipipe_backends', 'backends' => $list];
			}
			header('Content-Type: application/json');
			echo json_encode($out);
			exit;
		}
		if ( $api === 'flexipipe_languages' ) {
			$cmd = escapeshellarg($py) . ' -m flexipipe info languages --output-format json 2>&1';
			$json = shell_exec($cmd);
			$data = $json ? json_decode(trim($json), true) : null;
			$out = ['ok' => false, 'api' => 'flexipipe_languages', 'languages' => []];
			if ( is_array($data) && !empty($data['languages']) ) {
				$list = [];
				foreach ( $data['languages'] as $e ) {
					if ( !is_array($e) ) continue;
					$code = $e['iso_639_1'] ?? $e['iso_639_3'] ?? null;
					$name = $e['name'] ?? $code;
					if ( $code !== null && $code !== '' ) $list[] = ['code' => $code, 'name' => $name];
				}
				usort($list, function ($a, $b) { return strcasecmp($a['name'], $b['name']); });
				$out = ['ok' => true, 'api' => 'flexipipe_languages', 'languages' => $list];
			}
			header('Content-Type: application/json');
			echo json_encode($out);
			exit;
		}
		if ( $api === 'flexipipe_models' ) {
			$lang = isset($_GET['language']) ? trim((string)$_GET['language']) : '';
			$out = ['ok' => false, 'api' => 'flexipipe_models', 'models' => []];
			if ( $lang !== '' ) {
				$cmd = escapeshellarg($py) . ' -m flexipipe info models --language ' . escapeshellarg($lang) . ' --output-format json 2>&1';
				$json = shell_exec($cmd);
				$data = $json ? json_decode(trim($json), true) : null;
				if ( is_array($data) && isset($data['models']) ) {
					$list = [];
					foreach ( $data['models'] as $e ) {
						if ( !is_array($e) || empty($e['backend']) || empty($e['model']) ) continue;
						$list[] = [
							'backend' => (string)$e['backend'],
							'model'   => (string)$e['model'],
							'label'   => (string)$e['backend'] . ' / ' . (string)$e['model'],
						];
					}
					$out = ['ok' => true, 'api' => 'flexipipe_models', 'models' => $list];
				}
			}
			header('Content-Type: application/json');
			echo json_encode($out);
			exit;
		}
		if ( $api === 'flexipipe_profiles' ) {
			$out = [
				'ok' => true,
				'api' => 'flexipipe_profiles',
				'datasets' => flexipipe_list_profile_items('datasets'),
				'applications' => flexipipe_list_profile_items('applications'),
				'manifests' => flexipipe_list_profile_items('manifests'),
			];
			header('Content-Type: application/json');
			echo json_encode($out);
			exit;
		}
	}

	if ( $act == "upgrade" ) {
		shell_exec("FLEXIPIPE_QUIET_INSTALL=1 $venv->venvdir/bin/python -m pip install --upgrade git+https://github.com/ufal/flexipipe.git >> tmp/venv-install.log 2>&1 &"); // Install in the background
	} else if ( $act == "modinst" ) {
		$instmod = $_GET['module'];
		$venv->installmod($instmod, true); // Install in the background
	};
		
	if ( $act == "save" ) {
		# Save flexipipe defaults to this project's Resources/settings.xml (same structure EasyCorp and flexipipe info teitok use).
		$setpath = "Resources/settings.xml";
		if ( !file_exists($setpath) || !is_writable($setpath) ) {
			$maintext .= "<p class=wrong>Cannot save: $setpath not found or not writable.</p>";
		} else {
			$xml = simplexml_load_file($setpath, null, LIBXML_NOERROR | LIBXML_NOWARNING);
			if ( $xml === false ) {
				$maintext .= "<p class=wrong>Failed to load settings.</p>";
			} else {
				if ( !isset($xml->defaults) ) $xml->addChild("defaults");
				$defaults = $xml->defaults;
				# Default language (empty = autoselect)
				$defaults["lang"] = isset($_POST["default_lang"]) ? trim((string)$_POST["default_lang"]) : "";
				# Default backend/model (global)
				if ( !isset($defaults->flexipipe) ) $defaults->addChild("flexipipe");
				$fp = $defaults->flexipipe;
				$fp["backend"] = isset($_POST["default_backend"]) ? trim((string)$_POST["default_backend"]) : "";
				$fp["model"]   = isset($_POST["default_model"])   ? trim((string)$_POST["default_model"])   : "";
				# Flexipipe-specific options (TEITOK settings.xml defaults/flexipipe)
				$fp["download_model"]   = ( isset($_POST["download_model"]) && trim((string)$_POST["download_model"]) === "yes" ) ? "yes" : "no";
				$fp["known_tags_only"]   = ( isset($_POST["known_tags_only"]) && trim((string)$_POST["known_tags_only"]) === "yes" ) ? "yes" : "no";
				$fp["use_raw_text"]      = ( isset($_POST["use_raw_text"]) && trim((string)$_POST["use_raw_text"]) === "yes" ) ? "yes" : "no";
				# Per-language models: remove existing language children then add from POST
				$toRemove = [];
				foreach ( $fp->language as $lang ) $toRemove[] = $lang;
				foreach ( $toRemove as $lang ) {
					$dom = dom_import_simplexml($lang);
					if ( $dom && $dom->parentNode ) $dom->parentNode->removeChild($dom);
				}
				if ( !empty($_POST["lang_code"]) && is_array($_POST["lang_code"]) ) {
					foreach ( $_POST["lang_code"] as $i => $code ) {
						$code = trim((string)$code);
						if ( $code === "" ) continue;
						$be = isset($_POST["lang_backend"][$i]) ? trim((string)$_POST["lang_backend"][$i]) : "";
						$mo = isset($_POST["lang_model"][$i])   ? trim((string)$_POST["lang_model"][$i])   : "";
						$lang = $fp->addChild("language");
						$lang["code"] = $code;
						$lang["backend"] = $be;
						$lang["model"] = $mo;
					}
				}
				if ( !file_exists("backups") ) @mkdir("backups", 0755, true);
				$buname = "settings-" . date("Ymd") . ".xml";
				if ( !file_exists("backups/$buname") ) copy($setpath, "backups/$buname");
				$written = $xml->asXML();
				if ( $written !== false ) file_put_contents($setpath, $written);
				$maintext .= "<p>Settings saved. <a href='index.php?action=$action'>Back to Flexipipe</a></p>";
				print "<script language=Javascript>setTimeout(function(){ top.location='index.php?action=$action'; }, 1500);</script>";
			}
		}
	} else if ( $act == "models" ) {

		$maintext .= "<h1>NLP Pipeline: Flexipipe</h1>";

		$maintext .= "<h2>Backends</h2>
			<p>Greyed-out backends can be installed, but are not currently locally available.</p>";
		$cmd = '-m flexipipe info backends --output-format json';
		$rjson = $venv->exec($cmd);
		$blist = $rjson ? json_decode(trim($rjson)) : null;
		$bel = array();
		$maintext .= "<table data-sortable id=rollovertable class='fp-backends'><thead><tr><th>Name<th>Description<th>Status</thead><tbody>";
		if ( is_object($blist) && !empty($blist->backends) ) {
			foreach ( $blist->backends as $bitem ) {
				$dis = ( isset($bitem->available) && $bitem->available === false ) ? " style='color: #888;'" : "";
				if ( isset($bitem->available) && $bitem->available === true ) $bel[$bitem->backend . ""] = 1;
				$name = isset($bitem->name) ? htmlspecialchars($bitem->name) : ( isset($bitem->backend) ? htmlspecialchars($bitem->backend) : '' );
				if ( !empty($bitem->url) ) $name = "<a href='" . htmlspecialchars($bitem->url) . "' target='_blank' rel='noopener'>$name</a>";
				$desc = isset($bitem->description) ? htmlspecialchars($bitem->description) : '';
				$status = isset($bitem->status) ? htmlspecialchars($bitem->status) : '';
				$installLink = '';
				if ( !empty($bitem->install_hint) ) {
					if ( preg_match('/^\s*pip\s+install\s+(.+)$/i', $bitem->install_hint, $m) ) {
						$module = trim($m[1], " \t\"'");
						$installLink = " (<a href='index.php?action=$action&act=modinst&module=" . rawurlencode($module) . "'>install " . htmlspecialchars($module) . "</a>)";
					} else {
						$installLink = " (" . htmlspecialchars($bitem->install_hint) . ")";
					}
				}
				$maintext .= "<tr $dis><td>$name<td>$desc<td>$status$installLink";
			}
		}
		$maintext .= "</tbody></table>";
			
		$maintext .= "<h2>Downloadable and available models</h2>
			<p>Greyed-out models are not currently locally available, but can be downloaded if the backend is installed.
				You can test the models <a href='index.php?action=$action&act=tag'>here</a>, or by clicking on the model name.</p>";

		$cmd = '-m flexipipe info models'; # List the models
		$rjson = $venv->exec($cmd);
		$mlist = json_decode($rjson);
		
		$maintext .= "<table data-sortable id=rollovertable><thead><tr><th>Backend<th>Model<th>Language (ISO)<th>Language<th>Description</thead><tbody>";
		foreach ( $mlist->models as $mitem ) {
			if ( !$bel[$mitem->backend] & !$_GET['show'] == "all" ) continue;
			$dis = ""; if ( isset($mitem->installed) && $mitem->installed === false ) $dis = "style='color: #aaaaaa;'";
			$model = "<a $dis href='index.php?action=$action&act=tag&backend=$mitem->backend&model=$mitem->model&language=$mitem->language_iso'>$mitem->model</a>";
			if ( !$bel[$mitem->backend] ) $model = $mitem->model;
			$maintext .= "<tr $dis><td>$mitem->backend<td>$model<td>$mitem->language_iso<td>$mitem->language_name<td>$mitem->description";
		};
		$maintext .= "</tbody></table>
			<hr><p><a href='index.php?action=$action'>back</a>";

		# Create a dataTable
		$maintext .= "<script language=Javascript src=\"https://cdnjs.cloudflare.com/ajax/libs/sortable/0.8.0/js/sortable.min.js\"></script>
		<script language=Javascript>Sortable.init(); document.getElementById('filecol').click();</script>
		<link rel=\"stylesheet\" href=\"https://github.hubspot.com/sortable/css/sortable-theme-bootstrap.css\">
		";

	} else if ( $act == "profiles" ) {

		$maintext .= "<h1>NLP Pipeline: Flexipipe profiles</h1>";
		$maintext .= "<p>Dataset, application, and manifest registries are defined in <code>settings.xml</code> under <code>&lt;flexipipe&gt;</code> and read via <code>getset(\"flexipipe/...\")</code>. Manifest items normally specify <code>backend</code>, <code>model</code>, <code>xpres</code>, and <code>recond</code> directly — no separate manifest file is required.</p>";
		$maintext .= flexipipe_profiles_table_html('datasets', 'Dataset profiles (training/convert)');
		$maintext .= flexipipe_profiles_table_html('applications', 'Application profiles (TEITOK writeback policy)');
		$maintext .= flexipipe_profiles_table_html('manifests', 'Model manifests (model identity + applicability)');
		$maintext .= "<p><a href='index.php?action=$action&act=train'>Train from dataset profile</a> · <a href='index.php?action=$action&act=tag'>Test tagging</a></p>";
		$maintext .= "<hr><p><a href='index.php?action=$action'>back</a>";

	} else if ( $act == "train" ) {

		$maintext .= "<h1>NLP Pipeline: Flexipipe training</h1>";
		$datasetItems = flexipipe_list_profile_items('datasets');
		if ( empty($datasetItems) ) {
			$maintext .= "<p class=wrong>No dataset profiles found. Add <code>&lt;flexipipe&gt;&lt;datasets&gt;&lt;item key=\"...\" path=\"...\"/&gt;</code> to settings.xml.</p>";
			$maintext .= "<p><a href='index.php?action=$action&act=profiles'>View profile registries</a></p>";
		} else if ( $_SERVER['REQUEST_METHOD'] === 'POST' && !empty($_POST['dataset']) ) {
			$datasetKey = trim((string)$_POST['dataset']);
			$backend = trim((string)($_POST['backend'] ?? 'flexitag'));
			$datasetPath = flexipipe_profile_path('datasets', $datasetKey);
			if ( !$datasetPath || !file_exists($datasetPath) ) {
				$maintext .= "<p class=wrong>Dataset profile not found or missing file: " . htmlspecialchars($datasetKey) . "</p>";
			} else {
				$refresh = !empty($_POST['refresh_splits']) ? ' --refresh-splits' : '';
				$convertCmd = "-m flexipipe convert --type treebank --config " . escapeshellarg($datasetPath) . $refresh;
				$trainCmd = "-m flexipipe train --config " . escapeshellarg($datasetPath) . " --backend " . escapeshellarg($backend);
				$maintext .= "<p>Running convert…</p><pre>" . htmlspecialchars($venv->exec($convertCmd)) . "</pre>";
				$maintext .= "<p>Running train…</p><pre>" . htmlspecialchars($venv->exec($trainCmd)) . "</pre>";
				$maintext .= "<p>Training finished. Correct more files in TEITOK, then train again with <em>Refresh splits from XML</em> or edit CoNLL-U manually.</p>";
			}
			$maintext .= "<p><a href='index.php?action=$action&act=train'>Train again</a> · <a href='index.php?action=$action&act=profiles'>Profiles</a></p>";
		} else {
			$maintext .= "<p>Iterative TEITOK workflow: manually correct files → convert/train → apply model → correct more → retrain.</p>";
			$maintext .= "<form method='post' action='index.php?action=$action&act=train'><table>";
			$maintext .= "<tr><th>Dataset profile<td><select name='dataset'><option value=''>[choose]</option>";
			foreach ( $datasetItems as $key => $item ) {
				$label = is_array($item) && !empty($item['label']) ? $item['label'] : $key;
				$maintext .= "<option value='" . htmlspecialchars($key) . "'>" . htmlspecialchars($label) . " (" . htmlspecialchars($key) . ")</option>";
			}
			$maintext .= "</select>";
			$maintext .= "<tr><th>Backend<td><select name='backend'><option value='flexitag'>flexitag</option><option value='udpipe1'>udpipe1</option></select>";
			$maintext .= "<tr><th>Refresh splits<td><label><input type=checkbox name=refresh_splits value=1> Re-convert TEITOK XML into ud_folder before training</label>";
			$maintext .= "</table><p><button type=submit>Convert + train</button></form>";
			$maintext .= "<p><a href='index.php?action=$action&act=profiles'>View all profiles</a></p>";
		}
		$maintext .= "<hr><p><a href='index.php?action=$action'>back</a>";

	} else if ( $act == "json" ) {
	
		$type = $_GET['type'];
		if ( $type == "languages" ) $cmd = '-m flexipipe info languages'; # List the models
		else exit;
		
		$rjson = $venv->exec($cmd);
		print $rjson; exit;
		
	} else if ( $act == "tag" ) {

		$getBackend = isset($_GET['backend']) ? htmlspecialchars($_GET['backend']) : '';
		$getModel   = isset($_GET['model'])   ? htmlspecialchars($_GET['model'])   : '';
		$getLang    = isset($_GET['language']) ? htmlspecialchars($_GET['language']) : '';

		$maintext .= "<h1>NLP Pipeline: flexiPipe</h1>
		
			<p>Here you can try out whether flexipipe is working correctly, and see the performance of the various models in action. 
			 To test, enter a text, or select the Example checkbox to try the first article of the UDHR. Then select the language of the text, 
			 the backend to use for the NLP tasks, and the output format, and then click process.
			 To see which models are available, see the <a href='index.php?action=$action&act=models'>model list</a>.
			 If an annotated text appears, flexiPipe is working correctly.
			
				<form id=ajaxForm enctype=\"multipart/form-data\">
				<table>
				<tr><th>Language: <td><select name=language id=langsel><option value=\"\">[choose]</option></select>
				<tr><th>Backend: <td><select name=backend id=backendsel><option value=''>[auto-select]</option></select>
				<tr><th>Model: <td><select name=model id=modelsel><option value=''>[choose language first]</option></select>
				<tr><th>Model download: <td><input type=checkbox value=1 name=dl> Download when needed
				<tr><th>Pretokenize: <td><input type=checkbox value=1 name=pretok> Override model tokenization by sending pretokenized data
				<tr><th>Example: <td><input type=checkbox value=1 name=udhr> Use UDHR example when no text is provided
				<tr><th>Output format : <td><select name=output><option value='conllu'>CoNLL-U</option><option value='teitok'>TEITOK</option></select>
				<tr><th>Gold CONLLU (optional): <td><input type=file name=gold_conllu accept='.conllu,.conll'> Upload a CONLLU file with corrected annotations to compare model output and see accuracy.
				<tr><th>Gold result view: <td><label><input type=radio name=gold_view value=raw checked> Raw model output</label> <label><input type=radio name=gold_view value=compare> Token-by-token comparison (gold vs model)</label>
				</table>
				<p><textarea name=text style='width: 100%; height: 100px;'></textarea>
				<p><button type=submit value=Process>Process</button>
				</form>
				
				<div id=response></div>
				<hr><p><a href='index.php?action=$action'>back</a>
				
				<script language=Javascript src='$jsurl/tokedit.js'></script>
				<script language=Javascript src='$jsurl/tokview.js'></script>
				<script>
				// Preselection from URL (tag?language=...&backend=...&model=...)
				const getlangiso = " . json_encode($getLang) . ";
				const getbackend = " . json_encode($getBackend) . ";
				const getmodel   = " . json_encode($getModel) . ";
				const actionUrl  = " . json_encode('index.php?action=' . $action) . ";
				var tagModelsCache = [];

				// Using Fetch API with async/await
				document.getElementById('ajaxForm').addEventListener('submit', async function(e) {
					e.preventDefault();
					
					const form = e.target;
					const submitBtn = form.querySelector('button[type=\"submit\"]');
					const originalText = submitBtn.textContent;
					const responseDiv = document.getElementById('response');
					
					// Show loading
					submitBtn.textContent = 'Submitting...';
					submitBtn.disabled = true;
					responseDiv.innerHTML = '<div style=\"color: blue;\">Submitting...</div>';
					
					try {
						const formData = new FormData(form);
						
						const response = await fetch('index.php?action=$action&act=process', {
							method: 'POST',
							body: formData,
						});
						
						const result = await response.text();
						const text = result.trim();
						
						if (response.ok) {
							if (text.startsWith('<?xml') || text.startsWith('<')) {
								responseDiv.innerHTML = `
									<style>teiHeader { display: none; };</style>
									<div id=mtxt>
										\${result}
									</div>
								`;
							} else {
								responseDiv.innerHTML = `
									<div style=\"color: green; padding: 10px; border: 1px solid green; border-radius: 5px;\">
										\${result}
									</div>
								`;
							};
							
							// Clear form on success
							// form.reset();
							
							// Or update specific fields
							// document.getElementById('name').value = '';
							
						} else {
							throw new Error(result.error || `HTTP \${response.status}`);
						}
						
					} catch (error) {
						responseDiv.innerHTML = `
							<div style=\"color: red; padding: 10px; border: 1px solid red; border-radius: 5px;\">
								<strong>Error:</strong> \${error.message}
							</div>
						`;
						
					} finally {
						// Reset button
						submitBtn.textContent = originalText;
						submitBtn.disabled = false;
					}
				});
				
				// AJAX-driven Language -> Backend -> Model (same APIs as elsewhere)
				fetch(actionUrl + '&api=flexipipe_languages')
				  .then(function(r) { return r.json(); })
				  .then(function(data) {
				    if (data && data.ok && Array.isArray(data.languages)) {
				      fill_languages(data.languages);
				      if (getlangiso) load_models_for_language(getlangiso);
				    }
				  })
				  .catch(function(e) { console.error('Error:', e); });

				function fill_languages(languages) {
					var langsel = document.getElementById('langsel');
					if (!langsel) return;
					langsel.innerHTML = '<option value=\"\">[choose]</option>';
					languages.sort(function(a, b) { return (a.name || '').localeCompare(b.name || ''); });
					languages.forEach(function(item) {
					  var opt = document.createElement('option');
					  opt.value = item.code || '';
					  opt.textContent = item.name || item.code || '';
					  if ((item.code || '') === getlangiso) opt.selected = true;
					  langsel.appendChild(opt);
					});
				}

				function load_models_for_language(langCode) {
					if (!langCode) {
					  tagModelsCache = [];
					  fill_backends([]);
					  fill_models([], '');
					  return;
					}
					fetch(actionUrl + '&api=flexipipe_models&language=' + encodeURIComponent(langCode))
					  .then(function(r) { return r.json(); })
					  .then(function(data) {
					    var models = (data && data.ok && Array.isArray(data.models)) ? data.models : [];
					    tagModelsCache = models;
					    fill_backends(models);
					    var backendSel = document.getElementById('backendsel');
					    var selBe = (backendSel && backendSel.value) ? backendSel.value : '';
					    fill_models(models, selBe);
					    if (langCode === getlangiso && (getbackend || getmodel)) {
					      if (backendSel && getbackend) backendSel.value = getbackend;
					      fill_models(models, getbackend || '');
					      var modelSel = document.getElementById('modelsel');
					      if (modelSel && getmodel) {
					        for (var i = 0; i < modelSel.options.length; i++) {
					          var o = modelSel.options[i];
					          if (o.value === getmodel && (o.getAttribute('data-backend') || '') === (getbackend || '')) {
					            modelSel.selectedIndex = i;
					            break;
					          }
					        }
					      }
					    }
					  })
					  .catch(function(e) { console.error('Error:', e); });
				}

				function fill_backends(models) {
					var sel = document.getElementById('backendsel');
					if (!sel) return;
					var backends = [];
					models.forEach(function(m) {
					  if (m.backend && backends.indexOf(m.backend) === -1) backends.push(m.backend);
					});
					sel.innerHTML = '<option value=\"\">[auto-select]</option>';
					backends.sort().forEach(function(be) {
					  var opt = document.createElement('option');
					  opt.value = be;
					  opt.textContent = be;
					  if (be === getbackend) opt.selected = true;
					  sel.appendChild(opt);
					});
				}

				function fill_models(models, filterBackend) {
					var sel = document.getElementById('modelsel');
					if (!sel) return;
					sel.innerHTML = filterBackend ? '<option value=\"\">[choose]</option>' : '<option value=\"\">[all]</option>';
					var list = models;
					if (filterBackend) list = models.filter(function(m) { return m.backend === filterBackend; });
					list.forEach(function(m) {
					  var opt = document.createElement('option');
					  opt.value = m.model || '';
					  opt.textContent = m.label || (m.backend + ' / ' + m.model);
					  opt.setAttribute('data-backend', m.backend || '');
					  if ((m.model || '') === getmodel && (m.backend || '') === getbackend) opt.selected = true;
					  sel.appendChild(opt);
					});
				}

				document.getElementById('langsel').addEventListener('change', function() {
				  load_models_for_language(this.value);
				});

				document.getElementById('backendsel').addEventListener('change', function() {
				  fill_models(tagModelsCache, this.value || '');
				});

				document.getElementById('modelsel').addEventListener('change', function() {
				  var opt = this.options[this.selectedIndex];
				  var be = opt && opt.getAttribute('data-backend');
				  var backendsel = document.getElementById('backendsel');
				  if (be !== null && be !== undefined && backendsel) backendsel.value = be || '';
				});
				</script>
						<script>
		var formdef = {
'form':{ 'key':'form',  'display':'Written form',  'color':'#990000',  'admin':'1',  'noshow':'1'}, 
'nform':{ 'key':'nform',  'display':'Normalized form',  'inherit':'form',  'color':'#990099'}};
		var tagdef = {
'lemma':{ 'key':'lemma',  'display':'Lemma'}, 
'upos':{ 'key':'upos',  'display':'POS tag (universal)'}, 
'xpos':{ 'key':'xpos',  'type':'pos',  'display':'POS tag (national)'}, 
'feats':{ 'key':'feats',  'display':'Morphosyntactic features',  'type':'udfeats'}, 
'head':{ 'key':'head',  'display':'Dependency Head',  'noshow':'1',  'type':'id'}, 
'deprel':{ 'key':'deprel',  'display':'Dependency Relation'}, 
};
		var wordinfo = true;
		var satts = {};
			var hlbar;
			var orgtoks = new Object();
			var attributelist = Array(\"form\",\"nform\",\"lemma\",\"upos\",\"xpos\",\"feats\",\"head\",\"deprel\",\"id\");
			
				var floatnotes = true;
attributenames['form'] = \"Written form\";  attributenames['nform'] = \"Normalized form\";  attributenames['lemma'] = \"Lemma\";  attributenames['upos'] = \"POS tag (universal)\";  attributenames['xpos'] = \"POS tag (national)\";  attributenames['feats'] = \"Morphosyntactic features\";  attributenames['head'] = \"Dependency Head\";  attributenames['deprel'] = \"Dependency Relation\";  
		</script>
";
	
	} else if ( $act == "process" ) {

		if ( $cid ) {
		
			# Process a TEITOK file (optional manifest/application keys from settings.xml)
		
			$filename = "xmlfiles/".$cid;
			$manifestKey = isset($_GET['manifest']) ? trim((string)$_GET['manifest']) : '';
			$applicationKey = isset($_GET['application']) ? trim((string)$_GET['application']) : '';
			if ( $applicationKey === '' && isset($_GET['profile']) ) {
				$applicationKey = trim((string)$_GET['profile']);
			}

			$xmlContent = file_exists($filename) ? file_get_contents($filename) : '';
			list($resolvedManifestKey, $manifestItem) = flexipipe_resolve_manifest($filename, $xmlContent, $manifestKey !== '' ? $manifestKey : null);
			$applicationPath = $applicationKey !== '' ? flexipipe_profile_path('applications', $applicationKey) : null;

			$cmd = flexipipe_build_process_cmd($filename, $manifestItem, $applicationPath, array('verbose' => true));
			$result = $venv->exec($cmd);
			$meta = '';
			if ( $resolvedManifestKey ) $meta .= " manifest=" . htmlspecialchars($resolvedManifestKey);
			if ( $applicationKey ) $meta .= " application=" . htmlspecialchars($applicationKey);
			print "<p>Flexipipe applied$meta:</p><pre>".htmlentities($result)."</pre>";
			print "<script>top.location='index.php?action=file&cid=$cid';</script>";
			exit;

		} else if ( !empty($_FILES['gold_conllu']['tmp_name']) && is_uploaded_file($_FILES['gold_conllu']['tmp_name']) ) {

			# Gold CONLLU uploaded: extract text, run model, compare and show accuracy
			$goldPath = $_FILES['gold_conllu']['tmp_name'];
			$goldContent = file_get_contents($goldPath);
			if ( $goldContent === false ) {
				print "<p class='wrong'>Could not read uploaded file.</p>";
				exit;
			}
			$parsed = flexipipe_parse_conllu($goldContent, FLEXIPIPE_GOLD_MAX_SENTENCES, FLEXIPIPE_GOLD_MAX_TOKENS);
			if ( empty($parsed['sentences']) || trim($parsed['text']) === '' ) {
				print "<p class='wrong'>Uploaded file does not look like valid CoNLL-U (no sentences found).</p>";
				exit;
			}
			if ( !is_dir('tmp') ) @mkdir('tmp', 0755, true);
			$textPath = 'tmp/gold_input_' . getmypid() . '.txt';
			file_put_contents($textPath, $parsed['text']);
			$cmd = "-m flexipipe process --input " . escapeshellarg($textPath) . " --output-format conllu";
			if ( !empty($_POST['language']) && $_POST['language'] !== '[choose]' ) $cmd .= " --language " . escapeshellarg($_POST['language']);
			if ( !empty($_POST['backend']) ) $cmd .= " --backend " . escapeshellarg($_POST['backend']);
			if ( !empty($_POST['model']) ) $cmd .= " --model " . escapeshellarg($_POST['model']);
			if ( !empty($_POST['dl']) ) $cmd .= " --download-model";
			if ( !empty($_POST['pretok']) ) $cmd .= " --pretokenize";
			$output = $venv->exec($cmd);
			@unlink($textPath);
			$modelParsed = flexipipe_parse_conllu($output);
			$cmp = flexipipe_compare_conllu($parsed['sentences'], $modelParsed['sentences']);
			$upos_pct = $cmp['upos_total'] > 0 ? round(100 * $cmp['upos_ok'] / $cmp['upos_total'], 1) : '-';
			$lemma_pct = $cmp['lemma_total'] > 0 ? round(100 * $cmp['lemma_ok'] / $cmp['lemma_total'], 1) : '-';
			$xpos_pct = $cmp['xpos_total'] > 0 ? round(100 * $cmp['xpos_ok'] / $cmp['xpos_total'], 1) : '-';
			$upos_feats_pct = $cmp['upos_feats_total'] > 0 ? round(100 * $cmp['upos_feats_ok'] / $cmp['upos_feats_total'], 1) : '-';
			$uas_pct = $cmp['uas_total'] > 0 ? round(100 * $cmp['uas_ok'] / $cmp['uas_total'], 1) : '-';
			$las_pct = $cmp['las_total'] > 0 ? round(100 * $cmp['las_ok'] / $cmp['las_total'], 1) : '-';
			$goldView = isset($_POST['gold_view']) && $_POST['gold_view'] === 'compare' ? 'compare' : 'raw';
			print "<style>.fp-gold-compare-table{border-collapse:collapse;margin:0.5rem 0;font-size:0.9em;}.fp-gold-compare-table th,.fp-gold-compare-table td{border:1px solid #ccc;padding:4px 6px;text-align:left;}.fp-gold-compare-table th{background:#f0f0f0;}.fp-gold-diff{background:#ffd0d0;}.fp-gold-sent-text{margin:0.75rem 0 0.25rem 0;}.fp-gold-sent-mismatch{color:#666;margin:0.5rem 0;}.fp-gold-warning{background:#fff3cd;border:1px solid #ffc107;padding:0.5rem 0.75rem;margin:0.5rem 0;}</style>";
			print "<h3>Accuracy vs gold CoNLL-U</h3>";
			print "<table border='1' cellpadding='6' cellspacing='0' style='border-collapse:collapse; margin:0.5rem 0;'>";
			print "<tr><th>Metric</th><th>Score</th><th>Count</th></tr>";
			print "<tr><td>UPOS</td><td>" . $upos_pct . "%</td><td>" . $cmp['upos_ok'] . " / " . $cmp['upos_total'] . "</td></tr>";
			print "<tr><td>Lemma</td><td>" . $lemma_pct . "%</td><td>" . $cmp['lemma_ok'] . " / " . $cmp['lemma_total'] . "</td></tr>";
			if ( $cmp['xpos_total'] > 0 ) {
				print "<tr><td>XPOS</td><td>" . $xpos_pct . "%</td><td>" . $cmp['xpos_ok'] . " / " . $cmp['xpos_total'] . "</td></tr>";
			}
			if ( $cmp['upos_feats_total'] > 0 ) {
				print "<tr><td>UPOS+Feats</td><td>" . $upos_feats_pct . "%</td><td>" . $cmp['upos_feats_ok'] . " / " . $cmp['upos_feats_total'] . "</td></tr>";
			}
			print "<tr><td>UAS (head)</td><td>" . $uas_pct . "%</td><td>" . $cmp['uas_ok'] . " / " . $cmp['uas_total'] . "</td></tr>";
			print "<tr><td>LAS (head+deprel)</td><td>" . $las_pct . "%</td><td>" . $cmp['las_ok'] . " / " . $cmp['las_total'] . "</td></tr>";
			print "</table>";
			$nSent = count($parsed['sentences']);
			$skipped = (int)$cmp['sentences_skipped'];
			if ( !empty($parsed['truncated']) ) {
				print "<p class='fp-gold-warning'><strong>Note:</strong> Large gold file limited to the first " . $nSent . " sentences / " . (int)$parsed['token_count'] . " tokens for comparison.</p>";
			}
			if ( $skipped > 0 ) {
				print "<p><small>Sentences skipped (token count mismatch): " . $skipped . "</small></p>";
				if ( $nSent > 0 && $skipped >= $nSent * 0.5 ) {
					print "<p class='fp-gold-warning'><strong>Note:</strong> Most sentences did not align. Mismatches should be rare — if many fail, something is wrong (e.g. gold and model use different tokenization, or the model was not run with the same input). Check that the gold file uses <code># text = ...</code> and that tokenization matches.</p>";
				}
			}
			if ( $goldView === 'compare' ) {
				print "<h3>Token-by-token comparison (gold → model)</h3>";
				$goldTexts = isset($parsed['sentence_texts']) ? $parsed['sentence_texts'] : [];
				$modelTexts = isset($modelParsed['sentence_texts']) ? $modelParsed['sentence_texts'] : [];
				print flexipipe_gold_compare_html($parsed['sentences'], $modelParsed['sentences'], $goldTexts, $modelTexts);
			} else {
				print "<h3>Model output (CoNLL-U)</h3><pre>" . htmlspecialchars($output) . "</pre>";
			}
			exit;
		} else if ( $_POST['text'] || $_GET['text'] || $_POST['udhr'] ) {

			# We are asked to process raw text

			if ( !$_POST ) $_POST = $_GET;
			$text = $_POST['text'];
			$cmd = "-m flexipipe process";
			if ( $_POST['text'] ) {
				file_put_contents("tmp/hashfile.txt", $text);
				$cmd .= "  --input 'tmp/hashfile.txt'";
			};
			if ( $_POST['language'] && $_POST['language'] != "[choose]" ) $cmd .= " --language ".$_POST['language'];
			if ( $_POST['backend'] ) $cmd .= " --backend ".$_POST['backend'];
			if ( $_POST['model'] ) $cmd .= " --model ".$_POST['model'];
			if ( $_POST['output'] ) $cmd .= " --output-format ".$_POST['output'];
			if ( $_POST['dl'] ) $cmd .= " --download-model";
			if ( $_POST['pretok'] ) $cmd .= " --pretokenize";
			if ( $_POST['udhr'] && !$_POST['text'] ) $cmd .= " --example udhr";
			# print "<p>Command: $cmd";
			$output = $venv->exec($cmd);
			if ( $_POST['output'] == "conllu" ) print "<pre>$output</pre>";
			else print $output;
			exit;

		};
		exit;
	
	} else if ( !$venv->checkmod("flexipipe") ) {
		$maintext .= "<h1>NLP Pipeline: Flexipipe</h1>";

		if ( $act == "install" ) {
			shell_exec("FLEXIPIPE_QUIET_INSTALL=1 $venv->venvdir/bin/python -m pip install git+https://github.com/ufal/flexipipe.git > /dev/null 2>&1 &"); // Install in the background
			print "<p>installation started - reloading</p>
				<script>top.location = 'index.php?action=$action&act=waitinstall';</script>";
		} else if ( $act == "waitinstall" ) {
			$maintext .= "<p>Flexipipe is being installed in $venv->venvdir - which might take a while. If this message keeps appearing, contact IT to manually install flexipipe";
		} else {
			$maintext .= "<p>The Flexipipe NLP pipeline relies on <a href='https://github.com/ufal/flexipipe'>flexipipe</a> being installed in your local Python environment (<a href='indx.php?action=$action&act=install'>install</a>)";
		};
		
	} else {
		# Suppress [VENV] debug lines in output (venv_debug appends to $maintext when $debug is set)
		$save_debug = isset($debug) ? $debug : null;
		$debug = 0;

		$maintext .= "<h1>NLP Pipeline: Flexipipe</h1>";

		$tagurl = "index.php?action=$action&act=tag";
		$maintext .= "<style>.flexipipe-test-promo{background:#f0f7ff;border:1px solid #b8d4e8;border-radius:6px;padding:0.6rem 1rem;margin:0.5rem 0;}.flexipipe-test-link{font-weight:600;color:#0066b3;}</style>";
		$maintext .= "<p class='flexipipe-test-promo'><a href='" . htmlspecialchars($tagurl) . "' class='flexipipe-test-link'>Try out Flexipipe &amp; compare models</a> – test whether Flexipipe is working and see which model works best for your data. <a href='index.php?action=$action&act=profiles'>Profiles</a> · <a href='index.php?action=$action&act=train'>Train</a></p>";

		$maintext .= "<p>In TEITOK, Natural Language Processing (NLP) is by default done using <a href='https://github.com/ufal/flexipipe'>Flexipipe</a>, a flexible NLP pipeline architecture that can run NLP in a variety of different tools (backends), 
			which has been designed to integrated both the Universal Dependency pipelines and TEITOK. Here you can define how Flexipipe is used in TEITOK. To annotate actual files, go to a file, select <i>file manangement</i>, from where you can run 
			the NLP pipeline directly on the file. The language to use for the NLP pipeline can be set in the file metadata (language in the teiHeader), or in the settings of the project itself. When no language is provided anywhere, flexipipe will 
			attempt to automatically determine the language of the file, which is always less certain than providing the language by hand.</p>";

		$cmd = '-m flexipipe config --show'; # Show the configurations from the flexipipe home (/home/git/.flexipipe)
		$rjson = $venv->exec($cmd);
		$conf = json_decode($rjson);
		
		$cmd = '-m flexipipe info teitok'; # Show the configuaration read from TEITOK (settings.xml / cqpsettings.xml)
		$rjson = $venv->exec($cmd);
		$tconf = json_decode($rjson);
		
		$defbackend = getset("flexipipe/backend", $conf->defaults->backend); 
		$defmodel   = getset("defaults/flexipipe/model", "");
		if ( !$defmodel && is_object($conf->defaults) && isset($conf->defaults->model) ) $defmodel = $conf->defaults->model;
		$download_model = getset("flexipipe/download_model", $conf->defaults->download_model); if ( !$download_model ) $download_model = "no";
		$known_tags_only = getset("flexipipe/known_tags_only", $tconf->known_tags_only); if ( !$known_tags_only ) $known_tags_only = "no";
		$use_raw_text = getset("flexipipe/use_raw_text", $tconf->use_raw_text); if ( !$use_raw_text ) $use_raw_text = "no";
		$deflang = getset("defaults/language", $tconf->default_language);
		if ( !$deflang && isset($tconf->default_language) ) $deflang = $tconf->default_language;
		if ( !$deflang ) $deflang = "";

		$maintext .= "<h2>Default settings (used when no language/model are passed)</h2>
			<p>When you do not pass a language or model to flexipipe (e.g. from EasyCorp with autoselect and no model), flexipipe uses these definitions. Leave default language empty for auto-detect.</p>
			<form method='post' action='index.php?action=$action&act=save' id='flexipipe-defaults-form'>
			<table>
			<tr><th>Default language<td><select name='default_lang' id='fp-default-lang'><option value=''>Autoselect</option></select><td><i style='color: #aaaaaa'>Default language (ISO code). Empty = flexipipe will detect language when not provided.</i>
			<tr><th>Default model<td><select id='fp-default-model'><option value=''>— Use default —</option></select><input type='hidden' name='default_backend' id='fp-default-backend' value='" . htmlspecialchars($defbackend) . "'><input type='hidden' name='default_model' id='fp-default-model-hidden' value='" . htmlspecialchars($defmodel) . "'><td><i style='color: #aaaaaa'>Preferred model when default language is set. Select language first to load models.</i>
			<tr><th>Auto-download models<td><select name='download_model'><option value='no'" . ( $download_model === 'yes' ? '' : " selected" ) . ">No</option><option value='yes'" . ( $download_model === 'yes' ? " selected" : '' ) . ">Yes</option></select><td><i style='color: #aaaaaa'>Allow flexipipe to download missing models when needed.</i>
			<tr><th>Only known tags<td><select name='known_tags_only'><option value='no'" . ( $known_tags_only === 'yes' ? '' : " selected" ) . ">No</option><option value='yes'" . ( $known_tags_only === 'yes' ? " selected" : '' ) . ">Yes</option></select><td><i style='color: #aaaaaa'>Only output tags that are in the TEITOK tag set (recommended for TEITOK).</i>
			<tr><th>Use raw text<td><select name='use_raw_text'><option value='no'" . ( $use_raw_text === 'yes' ? '' : " selected" ) . ">No</option><option value='yes'" . ( $use_raw_text === 'yes' ? " selected" : '' ) . ">Yes</option></select><td><i style='color: #aaaaaa'>Send raw text to the backend (recommended for TEITOK; use when no tokens exist yet).</i>
			</table>
			<p><strong>Per-language default models</strong> – when a language is provided or detected, flexipipe uses the model for that language below (if set), otherwise the default backend/model above.</p>
			<table id='langmodels'><thead><tr><th>Language<th>Model</thead><tbody id='langmodels-tbody'>";

		$defmodels = getset("flexipipe/models", array());
		if ( !is_array($defmodels) ) $defmodels = array();
		$per_lang_js = array();
		foreach ( $defmodels as $lk => $dm ) {
			$be = is_array($dm) ? (isset($dm["backend"]) ? $dm["backend"] : "") : "";
			$mo = is_array($dm) ? (isset($dm["model"])   ? $dm["model"]   : "") : "";
			$maintext .= "<tr class='fp-lang-row'><td><select name='lang_code[]' class='fp-lang-select'><option value=''>—</option></select><td><select class='fp-model-select' data-backend-name='lang_backend[]' data-model-name='lang_model[]'><option value=''>— Use default —</option></select><input type='hidden' name='lang_backend[]' value='" . htmlspecialchars($be) . "'><input type='hidden' name='lang_model[]' value='" . htmlspecialchars($mo) . "'></tr>";
			$per_lang_js[] = array( 'code' => $lk, 'backend' => $be, 'model' => $mo );
		}
		# If no per-language rows yet, add one empty row so user can add at least one language
		if ( empty($defmodels) ) {
			$maintext .= "<tr class='fp-lang-row'><td><select name='lang_code[]' class='fp-lang-select'><option value=''>—</option></select><td><select class='fp-model-select' data-backend-name='lang_backend[]' data-model-name='lang_model[]'><option value=''>— Use default —</option></select><input type='hidden' name='lang_backend[]' value=''><input type='hidden' name='lang_model[]' value=''></tr>";
			$per_lang_js[] = array( 'code' => '', 'backend' => '', 'model' => '' );
		}
		# Template row for "Add language" (clone and append on add; names so clone submits)
		$maintext .= "<tr id='fp-lang-row-tpl' class='fp-lang-row fp-lang-row-tpl' style='display:none'><td><select name='lang_code[]' class='fp-lang-select'><option value=''>—</option></select><td><select class='fp-model-select' data-backend-name='lang_backend[]' data-model-name='lang_model[]'><option value=''>— Use default —</option></select><input type='hidden' name='lang_backend[]' value=''><input type='hidden' name='lang_model[]' value=''></tr>";
		$maintext .= "</tbody></table>
			<p><button type='button' id='fp-add-language'>Add language</button>
			<button type='submit'>Save flexipipe defaults</button></form>";

		$form_data = array(
			'default_lang'   => $deflang,
			'default_backend' => $defbackend,
			'default_model'  => $defmodel,
			'per_lang'       => $per_lang_js,
		);
		$maintext .= "<script type='text/javascript'>\nvar flexipipeFormData = " . json_encode($form_data) . ";\n</script>";
		$maintext .= "<script type='text/javascript'>\n" . file_get_contents(__DIR__ . '/../Scripts/flexipipe-form.js') . "\n</script>";
			
		$maintext .= "<h2>Backends</h2>
			<p>Greyed-out backends can be installed, but are not currently locally available. Install links run <code>pip install</code> in the background.</p>";
			
		$cmd = '-m flexipipe info backends --output-format json';
		$rjson = $venv->exec($cmd);
		$backends = $rjson ? json_decode(trim($rjson)) : null;
		
		$maintext .= "<table class='fp-backends'><thead><tr><th>Name<th>Description<th>Status</thead><tbody>";
		if ( is_object($backends) && !empty($backends->backends) ) {
			foreach ( $backends->backends as $be ) {
				$available = isset($be->available) && $be->available === true;
				$dis = $available ? '' : " style='color: #888;'";
				$name = isset($be->name) ? htmlspecialchars($be->name) : ( isset($be->backend) ? htmlspecialchars($be->backend) : '' );
				if ( !empty($be->url) ) {
					$url = htmlspecialchars($be->url);
					$name = "<a href='$url' target='_blank' rel='noopener'>$name</a>";
				}
				$desc = isset($be->description) ? htmlspecialchars($be->description) : '';
				$status = isset($be->status) ? htmlspecialchars($be->status) : ( $available ? 'available' : '' );
				$installLink = '';
				if ( !empty($be->install_hint) ) {
					$hint = $be->install_hint;
					if ( preg_match('/^\s*pip\s+install\s+(.+)$/i', $hint, $m) ) {
						$module = trim($m[1], " \t\"'");
						$enc = rawurlencode($module);
						$installLink = " (<a href='index.php?action=$action&act=modinst&module=$enc'>install " . htmlspecialchars($module) . "</a>)";
					} else {
						$installLink = " (" . htmlspecialchars($hint) . ")";
					}
				}
				$maintext .= "<tr$dis><td>$name<td>$desc<td>$status$installLink";
			}
		} else {
			$maintext .= "<tr><td colspan='3'>Could not load backends (run <code>flexipipe info backends --output-format json</code> to check).</td></tr>";
		}
		$maintext .= "</tbody></table>";

		$maintext .= "<p>A list of available models can be found <a href='index.php?action=$action&act=models'>here</a>.</p>";

		$version = $venv->exec("-m flexipipe --version");
		$maintext .= "<p style='color : #aaaaaa;'>Flexipipe version: $version</p>";

		if ( $save_debug !== null ) $debug = $save_debug;
	}


?>