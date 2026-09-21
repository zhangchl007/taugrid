// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

package cli

import (
	"bytes"
	"context"
	"io"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/spf13/cobra"

	"github.com/Azure/taugrid/core/status"
)

func recoveryExampleOptions(t *testing.T, scenario string) (unresolvedRunOptions, string) {
	t.Helper()
	file := filepath.Join("..", "..", "..", "examples", "checkpoint-recovery", "tau-"+scenario+".yaml")
	options, name, err := loadRunConfig(file)
	if err != nil {
		t.Fatal(err)
	}
	return options, name
}

func TestCheckpointRecoveryExamplesRenderDurableSingleGPUJobs(t *testing.T) {
	for scenario, retries := range map[string]int{"manual": 0, "auto": 1} {
		t.Run(scenario, func(t *testing.T) {
			options, name := recoveryExampleOptions(t, scenario)
			checkpoint := "/data/checkpoints/finetunes/recovery-" + scenario
			if name != "recovery-"+scenario || options.engine != "job" ||
				options.jobGPUs == nil || *options.jobGPUs != 1 ||
				options.maxRetries != retries || options.checkpointPath != checkpoint ||
				!slices.Contains(options.env, "RECOVERY_CHECKPOINT_DIR="+checkpoint) ||
				!slices.Contains(options.env, "RECOVERY_DEVICE=cuda") ||
				!slices.Equal(options.retryOn, []string{"Preempted", "Evicted"}) {
				t.Fatalf("example lost its recovery contract: name=%s options=%+v", name, options)
			}
			options.dryRun = "client"
			// A synthetic authoritative profile keeps this render test strictly offline.
			attachAuthoritativeProfileForTest(&options)
			target, err := resolveRunTarget(options, name)
			if err != nil {
				t.Fatal(err)
			}
			if resolvedJobRequestForTest(target) == nil {
				t.Fatal("recovery must use a single direct Job, not a multi-node RayJob")
			}
			parent := &cobra.Command{}
			parent.SetContext(context.Background())
			var output bytes.Buffer
			parent.SetOut(&output)
			parent.SetErr(io.Discard)
			if err := executeRunTarget(parent, target, "offline recovery render", options.experiment); err != nil {
				t.Fatal(err)
			}
			for _, want := range []string{
				"kind: Job", "name: " + name, "namespace: taugrid-default",
				"claimName: recovery-checkpoints", "mountPath: /data",
				"backoffLimit: 0", "restartPolicy: Never",
				"memory: 2Gi", "memory: 4Gi", "nvidia.com/gpu: 1",
				"RECOVERY_CHECKPOINT_DIR", checkpoint,
			} {
				if !strings.Contains(output.String(), want) {
					t.Errorf("rendered example missing %q", want)
				}
			}
		})
	}
}

func TestCheckpointRecoveryExampleBoundedRetryContract(t *testing.T) {
	for _, test := range []struct {
		name    string
		succeed bool
		unknown bool
		wantErr string
		retries int
	}{
		{name: "recovered", succeed: true, retries: 1},
		{name: "exhausted", wantErr: "exhausted all 1 retries", retries: 1},
		{name: "unknown", unknown: true, wantErr: "not in retry_on", retries: 0},
	} {
		t.Run(test.name, func(t *testing.T) {
			dispatch, name := recoveryExampleOptions(t, "auto")
			opts := retryLoopOptions{
				name: name, namespace: dispatch.namespace, maxRetries: dispatch.maxRetries,
				retryOn: dispatch.retryOn, checkpointPath: dispatch.checkpointPath,
				backoffInitial: dispatch.backoffInitial, backoffMax: dispatch.backoffMax,
			}
			waits, deletes, submissions, sleeps := 0, 0, 0, 0
			var replacementEnv []string
			err := retryLoopWithHooks(io.Discard, opts, retryHooks{
				waitForTerminal: func() (status.Snapshot, terminalState, error) {
					waits++
					if waits > 1 && test.succeed {
						return successSnapshot(), terminalSuccess, nil
					}
					snapshot := preemptedSnapshot()
					if test.unknown {
						snapshot.Pods = nil
					}
					return snapshot, terminalFailed, nil
				},
				prepareResubmit: func(attempt int, reason string) error {
					replacementEnv = appendRetryEnv(dispatch.env, opts.checkpointPath, attempt, opts.maxRetries, reason)
					return nil
				},
				deleteWorkload: func() error {
					if len(replacementEnv) == 0 {
						t.Fatal("delete occurred before replacement preparation")
					}
					deletes++
					return nil
				},
				resubmit: func(attempt int, reason string) error {
					submissions++
					for _, want := range []string{
						"TAU_RESUME_FROM=/data/checkpoints/finetunes/recovery-auto",
						"TAU_RETRY_ATTEMPT=1", "TAU_RETRY_MAX=1", "TAU_RETRY_REASON=Preempted",
					} {
						if !slices.Contains(replacementEnv, want) {
							t.Errorf("replacement missing %s", want)
						}
					}
					return nil
				},
				sleep: func(delay time.Duration) error {
					sleeps++
					if delay != 5*time.Second {
						t.Errorf("configured backoff = %s, want 5s", delay)
					}
					return nil
				},
			})
			if test.wantErr == "" && err != nil ||
				test.wantErr != "" && (err == nil || !strings.Contains(err.Error(), test.wantErr)) {
				t.Fatalf("error = %v, want %q", err, test.wantErr)
			}
			if deletes != test.retries || submissions != test.retries ||
				sleeps != test.retries || waits != test.retries+1 {
				t.Fatalf("unbounded/unexpected retry counts: waits=%d deletes=%d submissions=%d sleeps=%d",
					waits, deletes, submissions, sleeps)
			}
		})
	}
}
